"""Hermetic unit + integration tests for the Harvey LAB agent.

Zero network: the agent runs on a scripted Stirrup client, the rubric judge
is a stub, and task documents come from a fixture tree on disk. Covers the
data loader / splitters, the workspace file surface, the verdict parser +
all-pass aggregation, and one full agent → judge evaluation pass.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from harvey_lab.agent.agent import HarveyLabAgent, _render_task_template
from harvey_lab.agent.workspace import TaskWorkspace, task_source_from_dir
from harvey_lab.config import HarveyLabConfig
from harvey_lab.data.dataset import load_harvey_lab_records, practice_areas_for_records
from harvey_lab.data.task_splits import fixed_val_split, stratified_case_cap
from harvey_lab.evaluation.local_eval import evaluate_agent_on_records, evaluate_record
from harvey_lab.evaluation.scoring import (
    ALL_PASS_FIELD,
    CRITERION_PASS_RATE_FIELD,
    _parse_verdict,
    _scope_deliverables,
    score_all_pass,
)


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
    # Four+ practice areas so the splitter has whole areas to hold out.
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


def _build_agent(tasks_root: Path, body: str, **cfg: Any) -> HarveyLabAgent:
    return HarveyLabAgent(
        config=HarveyLabConfig(max_turns=5, **cfg),
        task_source=task_source_from_dir(tasks_root),
        model_factory=_scripted_model_factory(body),
    )


def _fee_judge(_desc: str, _title: str, _match: str, out: str) -> bool:
    return "$50,000" in out or "notes.txt" in out


# ─── data loader + splitters ──────────────────────────────────────────


def test_load_records(tasks_root: Path) -> None:
    records = load_harvey_lab_records(tasks_root)
    assert len(records) == 5
    rec = next(r for r in records if r.task_id == "contracts/t1")
    assert rec.practice_area == "contracts"
    assert rec.deliverable_names == ("memo.md",)
    assert rec.documents == ("notes.txt",)
    assert len(rec.criteria) == 2


def test_max_per_area_and_filter(tasks_root: Path) -> None:
    records = load_harvey_lab_records(tasks_root, practice_areas=["contracts"], max_per_area=1)
    assert len(records) == 1
    assert records[0].practice_area == "contracts"


def test_fixed_val_split_holds_out_whole_areas(tasks_root: Path) -> None:
    records = load_harvey_lab_records(tasks_root)
    train_pool, val_records, val_areas = fixed_val_split(records, n_val_areas=2, val_size=None, seed=0)
    train_areas = {r.practice_area for r in train_pool}
    assert len(val_areas) == 2
    assert not (train_areas & val_areas)  # disjoint: no practice area leaks
    assert {r.practice_area for r in val_records} == val_areas


def test_fixed_val_split_is_stable_across_seed_reruns(tasks_root: Path) -> None:
    records = load_harvey_lab_records(tasks_root)
    a = fixed_val_split(records, n_val_areas=2, val_size=None, seed=3)[2]
    b = fixed_val_split(records, n_val_areas=2, val_size=None, seed=3)[2]
    assert a == b


def test_stratified_case_cap_spreads_across_areas(tasks_root: Path) -> None:
    records = load_harvey_lab_records(tasks_root)
    capped = stratified_case_cap(records, 3, seed=0)
    assert len(capped) == 3
    # Round-robin picks distinct areas before deepening any one.
    assert len({r.practice_area for r in capped}) == 3


def test_practice_areas_for_records(tasks_root: Path) -> None:
    records = load_harvey_lab_records(tasks_root)
    assert practice_areas_for_records(records) == ["contracts", "corporate-ma", "litigation", "tax"]


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


def test_render_task_template_substitutes_and_falls_back() -> None:
    rendered = _render_task_template("{{instructions}}\n\n{{deliverables}}", instructions="do it", deliverables="- x")
    assert "do it" in rendered and "- x" in rendered
    # A template that drops a var still gets the raw value appended.
    dropped = _render_task_template("only instructions: {{instructions}}", instructions="I", deliverables="- d")
    assert "I" in dropped and "- d" in dropped


# ─── full agent → judge integration ──────────────────────────────────


def test_evaluate_record_all_pass(tasks_root: Path) -> None:
    agent = _build_agent(tasks_root, "Termination fee is $50,000, per notes.txt.")
    record = load_harvey_lab_records(tasks_root, practice_areas=["contracts"], max_per_area=1)[0]
    result = asyncio.run(evaluate_record(record=record, agent=agent, judge=_fee_judge))
    assert result["kind"] == "scored"
    assert result[ALL_PASS_FIELD] == 1.0
    assert result[CRITERION_PASS_RATE_FIELD] == 1.0


def test_evaluate_record_fail_when_judge_rejects(tasks_root: Path) -> None:
    agent = _build_agent(tasks_root, "irrelevant content")
    record = load_harvey_lab_records(tasks_root, practice_areas=["tax"], max_per_area=1)[0]
    result = asyncio.run(evaluate_record(record=record, agent=agent, judge=lambda *_: False))
    assert result[ALL_PASS_FIELD] == 0.0
    assert result[CRITERION_PASS_RATE_FIELD] == 0.0


def test_forward_uses_configured_prompts(tasks_root: Path) -> None:
    """The system prompt the LLM sees is the agent's configured prompt."""
    captured: dict[str, str] = {}

    class _CapturingClient:
        @property
        def model_slug(self) -> str:
            return "scripted/test"

        @property
        def max_tokens(self) -> int:
            return 100_000

        async def generate(self, messages: list[Any], tools: dict[str, Any]) -> Any:
            from stirrup.core.models import AssistantMessage, ToolCall

            if "system" not in captured and messages:
                captured["system"] = str(getattr(messages[0], "content", ""))
            return AssistantMessage(
                content="",
                tool_calls=[ToolCall(name="finish", arguments=json.dumps({"reason": "done"}), tool_call_id="tc-1")],
            )

    agent = HarveyLabAgent(
        config=HarveyLabConfig(max_turns=3),
        task_source=task_source_from_dir(tasks_root),
        model_factory=lambda *_: _CapturingClient(),
        system_prompt="MY-CUSTOM-SYSTEM-PROMPT",
    )
    record = load_harvey_lab_records(tasks_root, practice_areas=["contracts"], max_per_area=1)[0]
    asyncio.run(agent.forward(record=record))
    assert "MY-CUSTOM-SYSTEM-PROMPT" in captured.get("system", "")


def test_forward_succeeds_in_worker_thread(tasks_root: Path) -> None:
    """The eval may run cases off the main thread; the Stirrup session must not
    install a SIGINT handler (would raise "signal only works in main thread of
    the main interpreter")."""
    import threading

    agent = _build_agent(tasks_root, "Termination fee is $50,000, per notes.txt.")
    record = load_harvey_lab_records(tasks_root, practice_areas=["contracts"], max_per_area=1)[0]
    outcome: dict[str, Any] = {}

    def _worker() -> None:
        try:
            outcome["result"] = asyncio.run(evaluate_record(record=record, agent=agent, judge=_fee_judge))
        except BaseException as exc:  # noqa: BLE001 - surface to the assertion below
            outcome["error"] = exc

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()

    assert "error" not in outcome, f"evaluate_record raised off the main thread: {outcome.get('error')!r}"
    assert outcome["result"][ALL_PASS_FIELD] == 1.0


def test_evaluate_agent_on_records_aggregates(tasks_root: Path) -> None:
    agent = _build_agent(tasks_root, "Termination fee is $50,000, per notes.txt.")
    records = load_harvey_lab_records(tasks_root)
    report = asyncio.run(evaluate_agent_on_records(agent=agent, records=records, judge=_fee_judge, max_concurrency=2))
    assert report.num_cases == 5
    assert report.num_scored == 5
    assert report.num_errored == 0
    assert report.all_pass == 1.0
    assert report.criterion_pass_rate == 1.0


def test_evaluate_agent_contains_errors_and_excludes_unscoreable(tasks_root: Path) -> None:
    """One case erroring counts as 0 (deflates); an unscoreable case (empty
    rubric) is excluded from the denominator, not counted as a failure."""
    # Add an unscoreable task (no criteria).
    _write_task(tasks_root, "immigration", "t1", criteria=[], deliverables={"memo.md": "Memo"})
    records = load_harvey_lab_records(tasks_root)

    class _FlakyAgent(HarveyLabAgent):
        async def forward(self, *, record: Any) -> Any:  # type: ignore[override]
            if record.practice_area == "tax":
                raise RuntimeError("boom")  # forces that case to error
            return await super().forward(record=record)

    flaky = _FlakyAgent(
        config=HarveyLabConfig(max_turns=5),
        task_source=task_source_from_dir(tasks_root),
        model_factory=_scripted_model_factory("Termination fee is $50,000, per notes.txt."),
    )
    report = asyncio.run(evaluate_agent_on_records(agent=flaky, records=records, judge=_fee_judge, max_concurrency=2))
    assert report.num_cases == 6
    assert report.num_errored == 1  # the tax task
    assert report.num_unscoreable == 1  # the immigration task (empty rubric)
    assert report.num_scored == 4
    # denominator = scored (4) + errored (1) = 5; the errored case scores 0.
    assert report.all_pass == pytest.approx(4 / 5)
