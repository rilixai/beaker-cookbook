"""Hermetic unit + integration tests for the Harvey LAB agent.

Zero network: the agent runs on a scripted Stirrup client, the rubric judge
is a stub, and task documents come from a fixture tree on disk. Covers the
data loader (incl. nested task discovery), the frozen splits, the workspace
file surface, the batched verdict parser + all-pass aggregation, and one
full agent -> judge evaluation pass.
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
from harvey_lab.data.dataset import load_records, read_split
from harvey_lab.evaluation.run_eval import evaluate_agent_on_records, evaluate_record
from harvey_lab.evaluation.scoring import (
    ALL_PASS_FIELD,
    CRITERION_PASS_RATE_FIELD,
    _parse_batch_verdicts,
    _scope_deliverables,
    build_rubric_judge,
    score_rubric,
)


# ─── fixtures ─────────────────────────────────────────────────────────


def _write_task(root: Path, task_id: str, *, criteria: list[dict], deliverables: dict[str, str]) -> None:
    task_dir = root / task_id
    (task_dir / "documents").mkdir(parents=True, exist_ok=True)
    (task_dir / "documents" / "notes.txt").write_text(
        "Master Services Agreement between Acme and Beta. Termination fee: $50,000.",
        encoding="utf-8",
    )
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "title": f"task {task_id}",
                "work_type": "analyze",
                "tags": [task_id.split("/", 1)[0]],
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
    _write_task(root, "contracts/t1", criteria=crit, deliverables={"memo.md": "Memo"})
    _write_task(root, "contracts/t2", criteria=crit, deliverables={"memo.md": "Memo"})
    _write_task(root, "corporate-ma/t1", criteria=crit, deliverables={"memo.md": "Memo"})
    _write_task(root, "tax/t1", criteria=crit, deliverables={"memo.md": "Memo"})
    _write_task(root, "litigation/t1", criteria=crit, deliverables={"memo.md": "Memo"})
    # A nested task (larger areas nest sub-categories under the practice area).
    _write_task(root, "contracts/banking/deep", criteria=crit, deliverables={"memo.md": "Memo"})
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


def _fee_judge(_desc: str, criteria: list[dict], out: str) -> dict[str, bool]:
    """Batched stub judge: a criterion passes if the fee/source appears."""
    passed = "$50,000" in out or "notes.txt" in out
    return {str(c["id"]): passed for c in criteria}


def _all_pass_judge(_desc: str, criteria: list[dict], _out: str) -> dict[str, bool]:
    return {str(c["id"]): True for c in criteria}


def _reject_judge(_desc: str, criteria: list[dict], _out: str) -> dict[str, bool]:
    return {str(c["id"]): False for c in criteria}


# ─── data loader ──────────────────────────────────────────────────────


def test_load_records(tasks_root: Path) -> None:
    records = load_records(tasks_root)
    assert len(records) == 6
    rec = next(r for r in records if r.task_id == "contracts/t1")
    assert rec.practice_area == "contracts"
    assert rec.deliverable_names == ("memo.md",)
    assert rec.documents == ("notes.txt",)
    assert len(rec.criteria) == 2


def test_load_records_discovers_nested_tasks(tasks_root: Path) -> None:
    """Larger areas nest sub-categories; discovery must walk recursively."""
    records = load_records(tasks_root)
    nested = next(r for r in records if r.task_id == "contracts/banking/deep")
    assert nested.practice_area == "contracts"  # first path segment


def test_load_records_by_task_ids_preserves_order(tasks_root: Path) -> None:
    ids = ["tax/t1", "contracts/t1"]
    records = load_records(tasks_root, task_ids=ids)
    assert [r.task_id for r in records] == ids


# ─── frozen splits ────────────────────────────────────────────────────


def test_frozen_splits_are_disjoint_and_capped() -> None:
    train, val, test = read_split("train"), read_split("val"), read_split("test")
    assert len(val) == 100
    assert len(test) == 100
    assert len(train) > 0
    # Three-way disjoint, and every id is a practice-area-prefixed path.
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    assert all("/" in tid for tid in val)


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


# ─── batched verdict parsing + scoring ────────────────────────────────


def test_parse_batch_verdicts_maps_ids() -> None:
    reply = '{"verdicts": [{"id": "C1", "verdict": "pass"}, {"id": "C2", "verdict": "fail"}]}'
    assert _parse_batch_verdicts(reply, ["C1", "C2"]) == {"C1": True, "C2": False}


def test_parse_batch_verdicts_missing_ids_default_fail() -> None:
    # C2 omitted by the judge -> conservative FAIL (must not inflate).
    reply = 'noise {"verdicts": [{"id": "C1", "verdict": "pass"}]} trailing'
    assert _parse_batch_verdicts(reply, ["C1", "C2"]) == {"C1": True, "C2": False}


def test_parse_batch_verdicts_garbage_is_all_fail() -> None:
    assert _parse_batch_verdicts("no json here", ["C1"]) == {"C1": False}


def test_parse_batch_verdicts_tolerates_surrounding_prose_and_fences() -> None:
    # A code fence around the JSON, plus trailing prose that itself has braces:
    # the balanced-brace scan must recover the verdicts object, not over-capture.
    reply = (
        "Here is my grading:\n"
        '```json\n{"verdicts": [{"id": "C1", "verdict": "pass"}, {"id": "C2", "verdict": "pass"}]}\n```\n'
        "Note: formatting like {curly} should not break parsing."
    )
    assert _parse_batch_verdicts(reply, ["C1", "C2"]) == {"C1": True, "C2": True}


def test_build_rubric_judge_uses_injected_llm() -> None:
    """The judge sends all batched criteria in one call and parses the reply."""
    calls: list[list[dict]] = []

    def _llm(*, model: str, messages: list[dict]) -> str:
        calls.append(messages)
        return '{"verdicts": [{"id": "C1", "verdict": "pass"}, {"id": "C2", "verdict": "fail"}]}'

    judge = build_rubric_judge(model="stub/test", llm=_llm)
    criteria = [{"id": "C1", "title": "t", "match_criteria": "x"}, {"id": "C2", "title": "t", "match_criteria": "y"}]
    assert judge("task", criteria, "output") == {"C1": True, "C2": False}
    assert len(calls) == 1  # both criteria graded in a single batched call


def test_scope_deliverables_selects_named_only() -> None:
    scoped = _scope_deliverables(["memo.md"], {"memo.md": "A", "appendix.md": "B"}, max_chars=100)
    assert "A" in scoped and "B" not in scoped


def test_scope_deliverables_falls_back_to_all_when_unmatched() -> None:
    scoped = _scope_deliverables(["missing.md"], {"memo.md": "A"}, max_chars=100)
    assert "A" in scoped


def test_score_rubric_all_and_partial() -> None:
    criteria = [
        {"id": "C1", "title": "t", "match_criteria": "x", "deliverables": ["m.md"]},
        {"id": "C2", "title": "t", "match_criteria": "y", "deliverables": ["m.md"]},
    ]
    deliverables = {"m.md": "body"}
    result_all = score_rubric(
        criteria=criteria, deliverables=deliverables, task_description="t", judge=_all_pass_judge
    )
    assert result_all[ALL_PASS_FIELD] == 1.0
    assert result_all[CRITERION_PASS_RATE_FIELD] == 1.0
    assert result_all["passed"] == 2
    assert result_all["total_criteria"] == 2

    def _half(_desc: str, crits: list[dict], _out: str) -> dict[str, bool]:
        return {str(c["id"]): c["id"] == "C1" for c in crits}

    result_partial = score_rubric(criteria=criteria, deliverables=deliverables, task_description="t", judge=_half)
    assert result_partial[ALL_PASS_FIELD] == 0.0
    assert result_partial[CRITERION_PASS_RATE_FIELD] == 0.5


def test_score_rubric_batches_by_size() -> None:
    """Same-scope criteria are chunked into batches of ``batch_size``."""
    criteria = [{"id": f"C{i}", "title": "t", "match_criteria": "x", "deliverables": ["m.md"]} for i in range(5)]
    batch_sizes: list[int] = []

    def _judge(_desc: str, crits: list[dict], _out: str) -> dict[str, bool]:
        batch_sizes.append(len(crits))
        return {str(c["id"]): True for c in crits}

    result = score_rubric(
        criteria=criteria, deliverables={"m.md": "b"}, task_description="t", judge=_judge, batch_size=2
    )
    assert result[ALL_PASS_FIELD] == 1.0
    assert batch_sizes == [2, 2, 1]  # 5 criteria, one scope, chunks of 2


def test_score_rubric_empty_rubric_is_unscoreable() -> None:
    result = score_rubric(criteria=[], deliverables={}, task_description="t", judge=_all_pass_judge)
    assert result["total_criteria"] == 0


def test_render_task_template_substitutes_and_falls_back() -> None:
    rendered = _render_task_template("{{instructions}}\n\n{{deliverables}}", instructions="do it", deliverables="- x")
    assert "do it" in rendered and "- x" in rendered
    # A template that drops a var still gets the raw value appended.
    dropped = _render_task_template("only instructions: {{instructions}}", instructions="I", deliverables="- d")
    assert "I" in dropped and "- d" in dropped


# ─── full agent → judge integration ──────────────────────────────────


def test_evaluate_record_all_pass(tasks_root: Path) -> None:
    agent = _build_agent(tasks_root, "Termination fee is $50,000, per notes.txt.")
    record = load_records(tasks_root, task_ids=["contracts/t1"])[0]
    result = asyncio.run(evaluate_record(record=record, agent=agent, judge=_fee_judge))
    assert result["kind"] == "scored"
    assert result[ALL_PASS_FIELD] == 1.0
    assert result[CRITERION_PASS_RATE_FIELD] == 1.0


def test_evaluate_record_fail_when_judge_rejects(tasks_root: Path) -> None:
    agent = _build_agent(tasks_root, "irrelevant content")
    record = load_records(tasks_root, task_ids=["tax/t1"])[0]
    result = asyncio.run(evaluate_record(record=record, agent=agent, judge=_reject_judge))
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
    record = load_records(tasks_root, task_ids=["contracts/t1"])[0]
    asyncio.run(agent.forward(record=record))
    assert "MY-CUSTOM-SYSTEM-PROMPT" in captured.get("system", "")


def test_forward_succeeds_in_worker_thread(tasks_root: Path) -> None:
    """The eval may run cases off the main thread; the Stirrup session must not
    install a SIGINT handler (would raise "signal only works in main thread of
    the main interpreter")."""
    import threading

    agent = _build_agent(tasks_root, "Termination fee is $50,000, per notes.txt.")
    record = load_records(tasks_root, task_ids=["contracts/t1"])[0]
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
    records = load_records(tasks_root)
    report = asyncio.run(evaluate_agent_on_records(agent=agent, records=records, judge=_fee_judge, max_concurrency=2))
    assert report.num_cases == 6
    assert report.num_scored == 6
    assert report.num_errored == 0
    assert report.all_pass_rate == 1.0
    assert report.criterion_pass_rate == 1.0


def test_evaluate_agent_contains_errors_and_excludes_unscoreable(tasks_root: Path) -> None:
    """One case erroring counts as 0 (deflates); an unscoreable case (empty
    rubric) is excluded from the denominator, not counted as a failure."""
    # Add an unscoreable task (no criteria).
    _write_task(tasks_root, "immigration/t1", criteria=[], deliverables={"memo.md": "Memo"})
    records = load_records(tasks_root)

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
    assert report.num_cases == 7
    assert report.num_errored == 1  # the tax task
    assert report.num_unscoreable == 1  # the immigration task (empty rubric)
    assert report.num_scored == 5
    # denominator = scored (5) + errored (1) = 6; the errored case scores 0.
    assert report.all_pass_rate == pytest.approx(5 / 6)
