"""Hermetic unit + integration tests for the Harvey LAB recipe.

Zero network: the agent runs on a scripted Stirrup client, the rubric judge
is a stub, and task documents come from a fixture tree on disk. Covers the
data loader / splitters, the workspace file surface, the verdict parser +
all-pass aggregation, the scorer's unscoreable handling, and one full
``run_case`` → ``score_case`` pass through the real spec.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from harvey_lab.agent.agent import _render_task_template
from harvey_lab.agent.workspace import (
    TaskWorkspace,
    build_bundled_task_source,
    task_source_from_dir,
)
from harvey_lab.config import HarveyLabConfig
from harvey_lab.data.dataset import (
    HarveyLabDataLoader,
    attach_document_blobs,
    cases_from_records,
    load_harvey_lab_records,
    practice_areas_for_cases,
    record_to_row,
)
from harvey_lab.data.task_splits import fixed_val_split, stratified_case_cap
from harvey_lab.optimization.scoring import (
    ALL_PASS_FIELD,
    CRITERION_PASS_RATE_FIELD,
    HarveyLabScorer,
    _parse_verdict,
    _scope_deliverables,
    score_all_pass,
)
from harvey_lab.optimization.spec import build_harvey_lab_spec


# ─── fixtures ─────────────────────────────────────────────────────────


def _write_task(root: Path, area: str, slug: str, *, criteria: list[dict], deliverables: dict[str, str]) -> None:
    task_dir = root / area / slug
    (task_dir / "documents").mkdir(parents=True, exist_ok=True)
    (task_dir / "documents" / "notes.txt").write_text(
        "Master Services Agreement between Acme and Beta. Termination fee: $50,000.",
        encoding="utf-8",
    )
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "title": f"{area} task {slug}",
                "work_type": "analyze",
                "tags": [area],
                "instructions": "Summarize the termination fee and cite the source document.",
                "deliverables": deliverables,
                "criteria": criteria,
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def tasks_root(tmp_path: Path) -> Path:
    root = tmp_path / "tasks"
    crit = [
        {"id": "C1", "title": "States the fee", "match_criteria": "Mentions $50,000.", "deliverables": ["memo.md"]},
        {"id": "C2", "title": "Cites source", "match_criteria": "Cites notes.txt.", "deliverables": ["memo.md"]},
    ]
    # Four practice areas so the splitter has whole areas to hold out.
    _write_task(root, "contracts", "t1", criteria=crit, deliverables={"memo.md": "Memo"})
    _write_task(root, "contracts", "t2", criteria=crit, deliverables={"memo.md": "Memo"})
    _write_task(root, "corporate-ma", "t1", criteria=crit, deliverables={"memo.md": "Memo"})
    _write_task(root, "tax", "t1", criteria=crit, deliverables={"memo.md": "Memo"})
    _write_task(root, "litigation", "t1", criteria=crit, deliverables={"memo.md": "Memo"})
    return root


# ─── scripted Stirrup client ──────────────────────────────────────────


class _ScriptedClient:
    """A Stirrup ``LLMClient`` that replays a fixed tool-call script.

    Turn 1 writes a deliverable naming the fee + source; turn 2 finishes.
    Never touches the network.
    """

    def __init__(self, *, deliverable_body: str) -> None:
        self._deliverable_body = deliverable_body
        self._turn = 0

    @property
    def model_slug(self) -> str:
        return "scripted/test"

    @property
    def max_tokens(self) -> int:
        return 100_000

    async def generate(self, messages: list[Any], tools: dict[str, Any]) -> Any:
        from stirrup.core.models import AssistantMessage, ToolCall

        self._turn += 1
        if self._turn == 1:
            call = ToolCall(
                name="write_deliverable",
                arguments=json.dumps({"path": "memo.md", "content": self._deliverable_body}),
                tool_call_id="tc-1",
            )
        else:
            call = ToolCall(
                name="finish",
                arguments=json.dumps({"reason": "Deliverable written."}),
                tool_call_id="tc-2",
            )
        return AssistantMessage(content="", tool_calls=[call])


def _scripted_model_factory(body: str) -> Any:
    def _factory(_model: str, _temp: float, _max_tokens: int) -> Any:
        return _ScriptedClient(deliverable_body=body)

    return _factory


# ─── data loader + splitters ──────────────────────────────────────────


def test_load_records_and_row_roundtrip(tasks_root: Path) -> None:
    records = load_harvey_lab_records(tasks_root)
    assert len(records) == 5
    rec = next(r for r in records if r.task_id == "contracts/t1")
    assert rec.practice_area == "contracts"
    assert rec.deliverable_names == ("memo.md",)
    assert rec.documents == ("notes.txt",)
    assert len(rec.criteria) == 2
    # Row roundtrips through the JSONL parser back to an equivalent record.
    row = record_to_row(rec)
    reparsed = HarveyLabDataLoader().parse_row(row, _ctx())
    assert reparsed.task_id == rec.task_id
    assert reparsed.criteria == rec.criteria


def test_max_per_area_and_filter(tasks_root: Path) -> None:
    records = load_harvey_lab_records(tasks_root, practice_areas=["contracts"], max_per_area=1)
    assert len(records) == 1
    assert records[0].practice_area == "contracts"


def test_fixed_val_split_holds_out_whole_areas(tasks_root: Path) -> None:
    cases = cases_from_records(load_harvey_lab_records(tasks_root))
    train_pool, val_cases, val_areas = fixed_val_split(cases, n_val_areas=2, val_size=None, seed=0)
    train_areas = {c.group_key for c in train_pool}
    assert len(val_areas) == 2
    assert not (train_areas & val_areas)  # disjoint: no practice area leaks
    assert {c.group_key for c in val_cases} == val_areas


def test_fixed_val_split_is_stable_across_seed_reruns(tasks_root: Path) -> None:
    cases = cases_from_records(load_harvey_lab_records(tasks_root))
    a = fixed_val_split(cases, n_val_areas=2, val_size=None, seed=3)[2]
    b = fixed_val_split(cases, n_val_areas=2, val_size=None, seed=3)[2]
    assert a == b


def test_stratified_case_cap_spreads_across_areas(tasks_root: Path) -> None:
    cases = cases_from_records(load_harvey_lab_records(tasks_root))
    capped = stratified_case_cap(cases, 3, seed=0)
    assert len(capped) == 3
    # Round-robin picks distinct areas before deepening any one.
    assert len({c.group_key for c in capped}) == 3


def test_practice_areas_for_cases(tasks_root: Path) -> None:
    cases = cases_from_records(load_harvey_lab_records(tasks_root))
    assert practice_areas_for_cases(cases) == ["contracts", "corporate-ma", "litigation", "tax"]


# ─── workspace ────────────────────────────────────────────────────────


def test_workspace_read_write_search_collect(tmp_path: Path) -> None:
    ws = TaskWorkspace(tmp_path / "ws")
    (ws.documents_dir / "a.txt").write_text("hello WORLD line\nsecond line", encoding="utf-8")
    assert "documents/a.txt" in ws.list_files()
    assert "hello WORLD" in ws.read_document("a.txt")
    hits = ws.search_documents("world")
    assert hits and hits[0]["file"] == "documents/a.txt"
    ws.write_deliverable("out.md", "draft")
    assert ws.collect_deliverables() == {"out.md": "draft"}
    ws.edit_deliverable("out.md", "draft", "final")
    assert ws.collect_deliverables()["out.md"] == "final"


def test_workspace_rejects_escape(tmp_path: Path) -> None:
    ws = TaskWorkspace(tmp_path / "ws")
    with pytest.raises(ValueError):
        ws.write_deliverable("../escape.txt", "x")


# ─── verdict parsing + scoring ────────────────────────────────────────


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"verdict": "pass", "reasoning": "ok"}', True),
        ('{"verdict": "fail", "reasoning": "missing"}', False),
        ("Some reasoning.\nVERDICT: PASS", True),
        ("Some reasoning.\nVERDICT: NOT MET", False),
        ("The output does satisfy this.\nPASS", True),
        ("garbage with no verdict", False),
    ],
)
def test_parse_verdict(reply: str, expected: bool) -> None:
    assert _parse_verdict(reply) is expected


def test_scope_deliverables_selects_named_only() -> None:
    scoped = _scope_deliverables(["memo.md"], {"memo.md": "A", "appendix.md": "B"}, max_chars=100)
    assert "A" in scoped and "B" not in scoped


def test_scope_deliverables_falls_back_to_all_when_unmatched() -> None:
    scoped = _scope_deliverables(["missing.md"], {"memo.md": "A"}, max_chars=100)
    assert "A" in scoped


def test_score_all_pass_all_and_partial() -> None:
    criteria = [
        {"id": "C1", "title": "t", "match_criteria": "x", "deliverables": ["m.md"]},
        {"id": "C2", "title": "t", "match_criteria": "y", "deliverables": ["m.md"]},
    ]
    deliverables = {"m.md": "body"}
    result_all = score_all_pass(
        criteria=criteria, deliverables=deliverables, task_description="t", judge=lambda *_: True
    )
    assert result_all["all_pass"] == 1.0
    assert result_all["criterion_pass_rate"] == 1.0

    calls = {"n": 0}

    def _half(*_: Any) -> bool:
        calls["n"] += 1
        return calls["n"] == 1  # first passes, second fails

    result_partial = score_all_pass(criteria=criteria, deliverables=deliverables, task_description="t", judge=_half)
    assert result_partial["all_pass"] == 0.0
    assert result_partial["criterion_pass_rate"] == 0.5


def test_score_all_pass_empty_rubric_is_unscoreable() -> None:
    result = score_all_pass(criteria=[], deliverables={}, task_description="t", judge=lambda *_: True)
    assert result["n_total"] == 0


def test_scorer_reads_precomputed_fields() -> None:
    from rilixai import Case, CaseResult

    scorer = HarveyLabScorer()
    case = Case(input=None, case_id="c", ground_truth={})
    passed = asyncio.run(
        scorer.score_case(case=case, result=CaseResult(output={ALL_PASS_FIELD: 1.0, CRITERION_PASS_RATE_FIELD: 1.0}))
    )
    assert passed.field_scores[ALL_PASS_FIELD] == 1.0
    assert passed.objective == 1.0
    # objective weights only the dense field, but both are reported.
    assert set(passed.field_scores) == {ALL_PASS_FIELD, CRITERION_PASS_RATE_FIELD}


def test_scorer_unscoreable_none_omits_fields() -> None:
    from rilixai import Case, CaseResult

    scorer = HarveyLabScorer()
    case = Case(input=None, case_id="c", ground_truth={})
    score = asyncio.run(
        scorer.score_case(case=case, result=CaseResult(output={ALL_PASS_FIELD: None, CRITERION_PASS_RATE_FIELD: None}))
    )
    assert score.field_scores == {}


def test_render_task_template_substitutes_and_falls_back() -> None:
    rendered = _render_task_template("{{instructions}}\n\n{{deliverables}}", instructions="do it", deliverables="- x")
    assert "do it" in rendered and "- x" in rendered
    # A candidate that drops a var still gets the raw value appended.
    dropped = _render_task_template("only instructions: {{instructions}}", instructions="I", deliverables="- d")
    assert "I" in dropped and "- d" in dropped


# ─── full run_case → score_case integration ──────────────────────────


def test_end_to_end_run_case_all_pass(tasks_root: Path) -> None:
    from harvey_lab.agent.prompts import harvey_lab_seed_targets

    spec = build_harvey_lab_spec(
        config=HarveyLabConfig(max_turns=5),
        task_source=task_source_from_dir(tasks_root),
        model_factory=_scripted_model_factory("Termination fee is $50,000, per notes.txt."),
        judge=lambda _desc, _title, _match, out: "$50,000" in out or "notes.txt" in out,
    )
    records = load_harvey_lab_records(tasks_root, practice_areas=["contracts"], max_per_area=1)
    case = cases_from_records(records)[0]
    result = asyncio.run(spec.run_case(case=case, targets=harvey_lab_seed_targets(), runtime=None))
    assert result.output[ALL_PASS_FIELD] == 1.0
    score = asyncio.run(spec.scorer.score_case(case=case, result=result))
    assert score.objective == 1.0
    assert score.field_scores[ALL_PASS_FIELD] == 1.0


def test_end_to_end_run_case_fail_when_judge_rejects(tasks_root: Path) -> None:
    from harvey_lab.agent.prompts import harvey_lab_seed_targets

    spec = build_harvey_lab_spec(
        config=HarveyLabConfig(max_turns=5),
        task_source=task_source_from_dir(tasks_root),
        model_factory=_scripted_model_factory("irrelevant content"),
        judge=lambda *_: False,
    )
    records = load_harvey_lab_records(tasks_root, practice_areas=["tax"], max_per_area=1)
    case = cases_from_records(records)[0]
    result = asyncio.run(spec.run_case(case=case, targets=harvey_lab_seed_targets(), runtime=None))
    assert result.output[ALL_PASS_FIELD] == 0.0
    assert result.output[CRITERION_PASS_RATE_FIELD] == 0.0


def test_run_case_succeeds_in_worker_thread(tasks_root: Path) -> None:
    """The hosted optimizer runs cases off the main thread; the Stirrup
    session must not install a SIGINT handler (would raise "signal only works
    in main thread of the main interpreter")."""
    import threading

    from harvey_lab.agent.prompts import harvey_lab_seed_targets

    spec = build_harvey_lab_spec(
        config=HarveyLabConfig(max_turns=5),
        task_source=task_source_from_dir(tasks_root),
        model_factory=_scripted_model_factory("Termination fee is $50,000, per notes.txt."),
        judge=lambda _desc, _title, _match, out: "$50,000" in out or "notes.txt" in out,
    )
    records = load_harvey_lab_records(tasks_root, practice_areas=["contracts"], max_per_area=1)
    case = cases_from_records(records)[0]

    outcome: dict[str, Any] = {}

    def _worker() -> None:
        try:
            outcome["result"] = asyncio.run(spec.run_case(case=case, targets=harvey_lab_seed_targets(), runtime=None))
        except BaseException as exc:  # noqa: BLE001 - surface to the assertion below
            outcome["error"] = exc

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()

    assert "error" not in outcome, f"run_case raised off the main thread: {outcome.get('error')!r}"
    assert outcome["result"].output[ALL_PASS_FIELD] == 1.0


def test_embedded_documents_roundtrip_and_materialize(tasks_root: Path) -> None:
    """``--embed-documents`` bundles docs into the row; the bundled task
    source materializes them from the row with no network access."""
    record = load_harvey_lab_records(tasks_root, practice_areas=["contracts"], max_per_area=1)[0]
    embedded = attach_document_blobs(record, tasks_root)
    assert embedded.document_blobs, "expected at least one embedded document"

    # Row serialization carries the blobs and re-parses to the same payload.
    row = record_to_row(embedded)
    assert "document_blobs" in row
    reparsed = HarveyLabDataLoader().parse_row(row, _ctx())
    assert reparsed.document_blobs == dict(embedded.document_blobs)

    # The bundled source materializes documents from the row (repo/commit are
    # bogus — a fetch would fail, proving no network is used).
    source = build_bundled_task_source(repo="does/not-exist", commit="0" * 40)
    workspace = source(reparsed)
    listed = workspace.list_files()
    assert any("notes.txt" in name for name in listed)
    assert "$50,000" in workspace.read_document("notes.txt")


def test_partial_embed_fetches_missing_documents(
    tasks_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A document named in the row but absent from ``document_blobs`` must be
    fetched (not silently skipped). The fetch is stubbed so the test stays
    hermetic; it asserts the exact missing entry is requested and materialized."""
    import dataclasses

    from harvey_lab.agent import workspace as ws_mod

    fetched: list[str] = []

    def _fake_fetch(url: str) -> bytes:
        fetched.append(url)
        return b"unbundled-bytes"

    monkeypatch.setattr(ws_mod, "_fetch_bytes", _fake_fetch)

    record = load_harvey_lab_records(tasks_root, practice_areas=["contracts"], max_per_area=1)[0]
    embedded = attach_document_blobs(record, tasks_root)
    # Declare an extra document that has no embedded blob.
    partial = dataclasses.replace(embedded, documents=(*embedded.documents, "unbundled.txt"))

    source = build_bundled_task_source(repo="acme/repo", commit="0" * 40)
    workspace = source(partial)
    # Only the unbundled doc is fetched; embedded ones are materialized locally.
    assert len(fetched) == 1
    assert fetched[0].endswith("/documents/unbundled.txt")
    assert "unbundled-bytes" in workspace.read_document("unbundled.txt")
    assert "$50,000" in workspace.read_document("notes.txt")


def _ctx() -> Any:
    from rilixai import DatasetRowContext

    return DatasetRowContext(split="train", line_number=1)
