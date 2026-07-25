"""Consolidated APEX-Agents test suite (non-agent).

The FakeWorld-driven ReAct loop behavioral suite lives in
``test_agent.py``. This file consolidates the rest:

- dataset normalization + IB filter + WorldFiles readers
- rubric scoring + LLM judge (incl. parse-verdict calibration)
- world-level validation split + stratified cap
- the bounded-concurrency local evaluator (error containment,
  unscoreable exclusion) + the report helpers
- CLI selection / guard plumbing
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from apex_agents.agent.agent import ApexReActAgent
from apex_agents.config import ApexAgentsConfig
from apex_agents.data.dataset import (
    ApexAgentsRecord,
    RubricCriterion,
    filter_investment_banking,
    load_apex_agents_records,
    records_from_rows,
    world_ids_for_records,
)
from apex_agents.data.world_splits import (
    fixed_val_split,
    stratified_case_cap,
)
from apex_agents.evaluation.local_eval import evaluate_agent_on_records
from apex_agents.evaluation.report import eval_summary, heldout_subset_summary, write_json
from apex_agents.evaluation.scoring import (
    RUBRIC_FIELD,
    build_rubric_judge,
    score_rubric,
)
from apex_agents.tests.fake_world import FakeWorld


def _task_row(idx: int, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "task_id": f"ib-task-{idx}",
        "task_name": f"Build a DCF model {idx}",
        "domain": "Investment Banking",
        "prompt": f"Analyze the target company and produce a valuation memo {idx}.",
        "world_id": f"world-{idx % 3}",
        "task_input_files": [f"inputs/brief_{idx}.pdf"],
        "rubric": [
            {"verifier_id": "output_llm", "criteria": f"The memo states an enterprise value for task {idx}."},
            {"verifier_id": "output_llm", "criteria": f"The memo cites at least two comparable companies {idx}."},
        ],
    }
    row.update(overrides)
    return row


def _world_row(world_id: str) -> dict[str, Any]:
    return {
        "world_id": world_id,
        "world_name": f"Deal room {world_id}",
        "description": f"A data room for {world_id}.",
    }


# ─────────────────────────────────────────────────────────────────────
# Section 1: dataset + world file surface
# ─────────────────────────────────────────────────────────────────────


def test_records_from_rows_normalizes_task_dicts() -> None:
    records = records_from_rows([_task_row(0)])
    assert len(records) == 1
    record = records[0]
    assert isinstance(record, ApexAgentsRecord)
    assert record.task_id == "ib-task-0"
    assert record.domain == "Investment Banking"
    assert record.world_id == "world-0"
    assert len(record.rubric) == 2
    assert record.rubric[0].verifier_id == "output_llm"
    assert record.task_input_files == ("inputs/brief_0.pdf",)


def test_rubric_parses_json_string_and_skips_blank_criteria() -> None:
    row = _task_row(0, rubric=json.dumps([{"verifier_id": "output_llm", "criteria": "X"}]))
    rec = records_from_rows([row])[0]
    assert len(rec.rubric) == 1
    assert rec.rubric[0].criteria == "X"

    row2 = _task_row(0, rubric=[{"verifier_id": "output_llm"}, {"criteria": "  "}, {"criteria": "real"}])
    rec2 = records_from_rows([row2])[0]
    assert [c.criteria for c in rec2.rubric] == ["real"]
    assert rec2.rubric[0].verifier_id == "output_llm"


def test_records_from_rows_rejects_unsupported_items() -> None:
    with pytest.raises(TypeError):
        records_from_rows([object()])  # type: ignore[list-item]


def test_filter_ib_case_insensitive_and_world_ids_sorted_distinct() -> None:
    records = [
        ApexAgentsRecord("a", "", "Investment Banking", "p", "w1", (), (), {}),
        ApexAgentsRecord("b", "", "investment banking", "p", "w2", (), (), {}),
        ApexAgentsRecord("c", "", "Law", "p", "w3", (), (), {}),
    ]
    ib = filter_investment_banking(records)
    assert {r.task_id for r in ib} == {"a", "b"}

    assert world_ids_for_records(records_from_rows([_task_row(i) for i in range(6)])) == [
        "world-0",
        "world-1",
        "world-2",
    ]


def _install_fake_hf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tasks = [_task_row(i) for i in range(6)]
    tasks.append(_task_row(99, domain="Law"))
    worlds = [_world_row(f"world-{i}") for i in range(3)]
    tasks_path = tmp_path / "tasks_and_rubrics.json"
    worlds_path = tmp_path / "world_descriptions.json"
    tasks_path.write_text(json.dumps(tasks))
    worlds_path.write_text(json.dumps(worlds))

    def _fake_download(repo_id: str, filename: str, *, cache_dir: str | None) -> str:
        if filename == "tasks_and_rubrics.json":
            return str(tasks_path)
        if filename == "world_descriptions.json":
            return str(worlds_path)
        raise AssertionError(f"unexpected filename {filename}")

    monkeypatch.setattr(
        "apex_agents.data.dataset._hf_download",
        _fake_download,
    )


def test_load_apex_agents_records_filters_caps_and_attaches_world_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_hf(monkeypatch, tmp_path)
    records = load_apex_agents_records()
    # The Law row is filtered out.
    assert len(records) == 6
    assert all(r.domain == "Investment Banking" for r in records)
    assert records[0].world_name == "Deal room world-0"
    assert "data room" in records[0].world_description
    # Cap honored + file order preserved.
    capped = load_apex_agents_records(max_records=3)
    assert [r.task_id for r in capped] == ["ib-task-0", "ib-task-1", "ib-task-2"]
    # domain=None disables the IB filter.
    every = load_apex_agents_records(domain=None)
    assert len(every) == 7
    with pytest.raises(ValueError, match="max_records"):
        load_apex_agents_records(max_records=0)


def test_world_read_spreadsheet_exposes_all_sheets_and_targets_one(tmp_path: Path) -> None:
    """Regression: the computed model lives on a LATER tab.

    The old read_spreadsheet concatenated every sheet into one blob;
    downstream observation truncation then hid everything after sheet 1
    ("Executive Summary"), so the agent never saw the model outputs and
    was forced to rebuild the DCF (compounding error).
    """
    openpyxl = pytest.importorskip("openpyxl")
    from apex_agents.agent.world.world import WorldFiles

    wb = openpyxl.Workbook()
    cover = wb.active
    cover.title = "Executive Summary"
    cover["A1"] = "Project Titan cover narrative"
    model = wb.create_sheet("DCF Model")
    model["A1"] = "Equity Value"
    model["B1"] = 2804.69
    xlsx = tmp_path / "model.xlsx"
    wb.save(xlsx)

    w = WorldFiles(tmp_path)
    default = w.read_spreadsheet("model.xlsx")
    assert default.splitlines()[0].startswith("# Sheets:")
    assert "Executive Summary" in default and "DCF Model" in default
    assert "sheet=" in default
    assert "2804.69" not in default
    targeted = w.read_spreadsheet("model.xlsx", sheet="DCF Model")
    assert "# Sheet: DCF Model" in targeted
    assert "2804.69" in targeted
    bad = w.read_spreadsheet("model.xlsx", sheet="Nope")
    assert "ERROR" in bad and "DCF Model" in bad


def test_world_read_docx_extracts_paragraphs_and_tables(tmp_path: Path) -> None:
    """Legal worlds are .docx-heavy; without read_docx those cases were
    unwinnable (read_file returned binary gibberish)."""
    docx = pytest.importorskip("docx")
    from apex_agents.agent.world.world import WorldFiles

    d = docx.Document()
    d.add_paragraph("Section 8.2 — Indemnification.")
    t = d.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text = "Cap"
    t.rows[0].cells[1].text = "HK$3,125,000"
    d.save(tmp_path / "spa.docx")

    text = WorldFiles(tmp_path).read_docx("spa.docx")
    assert "Indemnification" in text
    assert "HK$3,125,000" in text


def test_world_read_xls_legacy_format(tmp_path: Path) -> None:
    """openpyxl is .xlsx-only; legacy .xls must route to xlrd."""
    xlwt = pytest.importorskip("xlwt", reason="xlwt only needed to author a .xls fixture")
    pytest.importorskip("xlrd")
    from apex_agents.agent.world.world import WorldFiles

    wb = xlwt.Workbook()
    ws = wb.add_sheet("Model")
    ws.write(0, 0, "EquityValue")
    ws.write(0, 1, 2804.69)
    wb.save(str(tmp_path / "legacy.xls"))

    out = WorldFiles(tmp_path).read_spreadsheet("legacy.xls")
    assert "# Sheets:" in out and "Model" in out
    assert "2804.69" in out


def test_world_read_pdf_routes_zip_headed_file(tmp_path: Path) -> None:
    """Files named .pdf that are actually ZIP/Office containers must return
    an actionable routing message, not an opaque pypdf failure."""
    from apex_agents.agent.world.world import WorldFiles

    (tmp_path / "survey.pdf").write_bytes(b"PK\x03\x04\x14\x00\x00\x00rest-of-zip")
    msg = WorldFiles(tmp_path).read_pdf("survey.pdf")
    assert "NOT a PDF" in msg
    assert "read_docx" in msg and "read_spreadsheet" in msg


# ─────────────────────────────────────────────────────────────────────
# Section 2: rubric scoring + LLM judge
# ─────────────────────────────────────────────────────────────────────


def test_score_rubric_with_stub_judge() -> None:
    rubric = [
        {"verifier_id": "output_llm", "criteria": "mentions EV"},
        {"verifier_id": "output_llm", "criteria": "mentions comps"},
        {"verifier_id": "output_llm", "criteria": "mentions risks"},
    ]

    def _judge(criterion: str, answer: str, task_prompt: str) -> bool:
        return "EV" in criterion and "EV" in answer

    assert score_rubric(rubric=rubric, answer="The EV is $10M.", task_prompt="t", judge=_judge) == 1 / 3
    assert score_rubric(rubric=[], answer="x", task_prompt="t", judge=lambda *_: True) == 0.0


def test_build_rubric_judge_parses_verdicts_and_handles_failures() -> None:
    # Plain string responses.
    calls: list[list[dict[str, str]]] = []

    def _fake_llm(*, model: str, messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return "MET" if "crit-0" in messages[0]["content"] else "NOT MET"

    judge = build_rubric_judge(model="stub/model", llm=_fake_llm)
    assert judge("crit-0", "answer", "task") is True
    assert judge("crit-1", "answer", "task") is False
    assert len(calls) == 2

    # litellm-shaped dict response.
    def _shaped_llm(*, model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "  not met  "}}]}

    assert build_rubric_judge(model="stub/model", llm=_shaped_llm)("c", "a", "t") is False

    # Ambiguous/unparseable replies → conservative NOT-MET.
    assert build_rubric_judge(model="stub/model", llm=lambda **_: "I am not sure either way.")("c", "a", "t") is False
    assert build_rubric_judge(model="stub/model", llm=lambda **_: "banana")("c", "a", "t") is False

    # Judge outage → conservative NOT-MET.
    def _broken_llm(**_: Any) -> str:
        raise RuntimeError("judge down")

    assert build_rubric_judge(model="stub/model", llm=_broken_llm)("c", "a", "t") is False


def test_parse_verdict_is_robust_to_verbose_reasoning_judges() -> None:
    """Regression: a verbose judge that concludes MET must parse as MET.

    The old bare-substring scan flipped replies like
    'MET — the figure is correct and there are no errors' to Not-met
    because it matched the 'no' in 'no errors' (NOT-MET checked first).
    That systematically depressed scores for reasoning/'thinking'
    judges (gemini-3.5-flash) vs terse ones (gpt-4.1).
    """
    from apex_agents.evaluation.scoring import _parse_verdict

    assert _parse_verdict("The value matches; there are no errors.\nVERDICT: MET") is True
    assert _parse_verdict("Reasoning: it fails the threshold.\nVERDICT: NOT MET") is False
    # The exact failure mode the old parser got wrong:
    assert _parse_verdict("MET — the figure is correct and there are no errors") is True
    assert _parse_verdict("This does not satisfy the criterion.\nVERDICT: NOT MET") is False
    assert _parse_verdict("verdict: met") is True
    assert _parse_verdict("VERDICT: NOT_MET") is False
    assert _parse_verdict("Draft: NOT MET. On reflection: VERDICT: MET") is True
    assert _parse_verdict("MET") is True
    assert _parse_verdict("  not met  ") is False
    # A last line that merely *starts* with a "MET…" word (METHOD /
    # METADATA / METRICS) must NOT be scored Met — the old
    # startswith("MET") broadening inflated rubric_pass_rate on these.
    assert _parse_verdict("METHODOLOGY looks sound") is False
    assert _parse_verdict("METADATA is complete") is False


# ─────────────────────────────────────────────────────────────────────
# Section 3: world-level validation split + stratified cap
# ─────────────────────────────────────────────────────────────────────


def _records(n_worlds: int, per_world: int = 3) -> list[ApexAgentsRecord]:
    return [
        ApexAgentsRecord(f"world-{w:02d}-{i}", "", "Investment Banking", "p", f"world-{w:02d}", (), (), {})
        for w in range(n_worlds)
        for i in range(per_world)
    ]


def test_fixed_val_split_constant_and_disjoint() -> None:
    records = _records(9, per_world=10)
    tp, val, vw = fixed_val_split(records, n_val_worlds=2, val_size=20, seed=0)
    assert len(vw) == 2
    assert {r.world_id for r in tp}.isdisjoint({r.world_id for r in val})
    assert {r.world_id for r in val} == vw
    assert len(val) == 20
    _, val2, vw2 = fixed_val_split(records, n_val_worlds=2, val_size=20, seed=0)
    assert [r.task_id for r in val2] == [r.task_id for r in val] and vw2 == vw
    assert len({r.world_id for r in val}) == 2


def test_stratified_cap_keeps_worlds_wide() -> None:
    pool = _records(9, per_world=10)
    strat = stratified_case_cap(pool, 9, mode="stratified", seed=0)
    assert len(strat) == 9 and len({r.world_id for r in strat}) == 9
    with pytest.raises(ValueError, match="must be 'stratified'"):
        stratified_case_cap(pool, 9, mode="frontslice", seed=0)
    assert stratified_case_cap(pool, None) == pool
    assert [r.task_id for r in stratified_case_cap(pool, 18, seed=0)] == [
        r.task_id for r in stratified_case_cap(pool, 18, seed=0)
    ]
    assert len({r.world_id for r in stratified_case_cap(pool, 18, seed=0)}) == 9


# ─────────────────────────────────────────────────────────────────────
# Section 4: the local batch evaluator + report helpers
# ─────────────────────────────────────────────────────────────────────


def _eval_record(task_id: str, world_id: str, *, rubric: tuple[RubricCriterion, ...]) -> ApexAgentsRecord:
    return ApexAgentsRecord(
        task_id=task_id,
        task_name="memo",
        domain="Investment Banking",
        prompt="State the EV.",
        world_id=world_id,
        rubric=rubric,
        task_input_files=(),
        raw_task={"task_id": task_id},
    )


def _scripted_model_factory(answer: str = "The EV is $5M.") -> Any:
    class _M:
        def complete(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                "content": "Submitting the memo.",
                "tool_calls": [
                    {"id": "c1", "name": "final_answer", "arguments": {"answer": answer, "status": "completed"}}
                ],
                "cost": 0.0,
            }

    return lambda _name, _temp: _M()


def _ev_judge(criterion: str, answer: str, task_prompt: str) -> bool:
    return "EV" in answer


def _build_agent(answer: str = "The EV is $5M.") -> ApexReActAgent:
    config = ApexAgentsConfig(max_steps=5)
    return ApexReActAgent(
        model_name=config.task_model,
        max_steps=config.max_steps,
        world_factory=lambda _record: FakeWorld({"brief.txt": "value"}),
        model_factory=_scripted_model_factory(answer),
    )


def test_evaluate_agent_on_records_aggregates() -> None:
    records = [
        _eval_record("ib-1", "world-a", rubric=(RubricCriterion("output_llm", "States an EV."),)),
        _eval_record("ib-2", "world-b", rubric=(RubricCriterion("output_llm", "States an EV."),)),
    ]
    report = asyncio.run(
        evaluate_agent_on_records(agent=_build_agent(), records=records, judge=_ev_judge, max_concurrency=2)
    )
    assert report.num_cases == 2
    assert report.num_scored == 2
    assert report.num_errored == 0
    assert report.rubric_pass_rate == 1.0
    assert {c["task_id"] for c in report.per_case} == {"ib-1", "ib-2"}


def test_evaluate_agent_contains_errors_and_excludes_unscoreable() -> None:
    """One task erroring counts as 0 (deflates) but does not abort the batch;
    an unscoreable task (empty rubric) is excluded from the denominator."""
    records = [
        _eval_record("ib-good", "world-a", rubric=(RubricCriterion("output_llm", "States an EV."),)),
        _eval_record("ib-bad", "world-b", rubric=(RubricCriterion("output_llm", "States an EV."),)),
        _eval_record("ib-empty", "world-c", rubric=()),
    ]

    class _FlakyAgent(ApexReActAgent):
        async def forward(self, *, record: ApexAgentsRecord) -> Any:
            if record.task_id == "ib-bad":
                raise RuntimeError("boom")
            return await super().forward(record=record)

    flaky = _FlakyAgent(
        model_name="scripted/test",
        max_steps=5,
        world_factory=lambda _record: FakeWorld({}),
        model_factory=_scripted_model_factory(),
    )
    report = asyncio.run(evaluate_agent_on_records(agent=flaky, records=records, judge=_ev_judge, max_concurrency=2))
    assert report.num_cases == 3
    assert report.num_errored == 1
    assert report.num_unscoreable == 1
    assert report.num_scored == 1
    # denominator = scored (1) + errored (1) = 2; the errored task scores 0.
    assert report.rubric_pass_rate == pytest.approx(0.5)
    errored = next(c for c in report.per_case if c["task_id"] == "ib-bad")
    assert errored["kind"] == "error"
    assert "RuntimeError: boom" in errored["error"]


def test_evaluate_agent_treats_a_failed_agent_run_as_an_error() -> None:
    """The agent reports a crash as an output with ``extra['error']`` rather
    than raising; that must land as an errored (0-scoring) case, ungraded."""

    def _exploding_factory(_name: str, _temp: float) -> Any:
        class _M:
            def complete(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
                raise TimeoutError("provider timed out")

        return _M()

    agent = ApexReActAgent(
        model_name="scripted/test",
        max_steps=5,
        world_factory=lambda _record: FakeWorld({}),
        model_factory=_exploding_factory,
    )

    def _judge(criterion: str, answer: str, task_prompt: str) -> bool:
        raise AssertionError("a failed agent run must never be graded")

    records = [_eval_record("ib-1", "world-a", rubric=(RubricCriterion("output_llm", "States an EV."),))]
    report = asyncio.run(evaluate_agent_on_records(agent=agent, records=records, judge=_judge, max_concurrency=1))
    assert report.num_errored == 1
    assert report.num_scored == 0
    assert report.rubric_pass_rate == 0.0
    assert report.per_case[0]["kind"] == "error"
    assert "provider timed out" in report.per_case[0]["error"]


def test_evaluate_agent_respects_max_concurrency() -> None:
    """No more than ``max_concurrency`` tasks may be in flight at once."""
    records = [
        _eval_record(f"ib-{i}", f"world-{i}", rubric=(RubricCriterion("output_llm", "States an EV."),))
        for i in range(6)
    ]
    in_flight = 0
    peak = 0

    class _TrackingAgent(ApexReActAgent):
        async def forward(self, *, record: ApexAgentsRecord) -> Any:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                await asyncio.sleep(0)
                return await super().forward(record=record)
            finally:
                in_flight -= 1

    agent = _TrackingAgent(
        model_name="scripted/test",
        max_steps=5,
        world_factory=lambda _record: FakeWorld({}),
        model_factory=_scripted_model_factory(),
    )
    report = asyncio.run(evaluate_agent_on_records(agent=agent, records=records, judge=_ev_judge, max_concurrency=2))
    assert report.num_cases == 6
    assert peak <= 2


def test_eval_summary_and_write_json_roundtrip(tmp_path: Path) -> None:
    records = [_eval_record("ib-1", "world-a", rubric=(RubricCriterion("output_llm", "States an EV."),))]
    report = asyncio.run(evaluate_agent_on_records(agent=_build_agent(), records=records, judge=_ev_judge))
    summary = eval_summary(report, split="all")
    assert summary["split"] == "all"
    assert summary["num_cases"] == 1
    assert summary[RUBRIC_FIELD] == 1.0

    path = tmp_path / "nested" / "eval_summary.json"
    write_json(path, summary)
    assert json.loads(path.read_text())[RUBRIC_FIELD] == 1.0


def test_heldout_subset_summary_counts_errors_and_drops_unscoreable() -> None:
    """The held-out mean mirrors the headline rule: scored + errored tasks are
    measurable (errors count as 0), unscoreable tasks and reserved validation
    worlds drop out."""
    per_case: list[dict[str, Any]] = [
        {"kind": "scored", "world_id": "w-clean", RUBRIC_FIELD: 1.0},
        {"kind": "error", "world_id": "w-clean", "error": "boom"},
        {"kind": "unscoreable", "world_id": "w-clean"},
        {"kind": "scored", "world_id": "w-val", RUBRIC_FIELD: 1.0},
    ]
    summary = heldout_subset_summary(per_case, {"w-val"})
    assert summary["excluded_world_ids"] == ["w-val"]
    # w-clean only: 1 scored (1.0) + 1 errored (0.0); unscoreable excluded.
    assert summary["num_heldout_cases"] == 2
    assert summary[f"{RUBRIC_FIELD}_heldout"] == pytest.approx(0.5)

    empty = heldout_subset_summary(per_case[3:], {"w-val"})
    assert empty["num_heldout_cases"] == 0 and empty[f"{RUBRIC_FIELD}_heldout"] is None


# ─────────────────────────────────────────────────────────────────────
# Section 5: CLI plumbing
# ─────────────────────────────────────────────────────────────────────


def test_no_network_guard_blocks_gated_dataset_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--no-network`` must refuse the gated HF dataset download.

    Regression: the guard originally only covered the world factory +
    judge, so record loading hit ``hf_hub_download`` before the guard was
    consulted and leaked the HF client's "gated repo" traceback.
    """
    import argparse

    from apex_agents import cli as apex_cli

    def _must_not_be_called(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("load_apex_agents_records was called despite --no-network")

    monkeypatch.setattr(apex_cli, "load_apex_agents_records", _must_not_be_called)

    args = argparse.Namespace(no_network=True, domain="Investment Banking", cache_dir=None)
    with pytest.raises(RuntimeError, match="Refusing to download the gated HF dataset"):
        apex_cli._load_all_records(args)


def test_cli_select_records_splits_and_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--split validation`` selects whole reserved worlds; ``--split all``
    returns everything and reports those worlds as excluded."""
    import argparse

    from apex_agents import cli as apex_cli

    pool = _records(4, per_world=3)
    monkeypatch.setattr(apex_cli, "_load_all_records", lambda _args: list(pool))

    base: dict[str, Any] = {
        "val_worlds": 1,
        "val_size": 0,
        "seed": 0,
        "test_size": None,
        "no_network": False,
    }
    val_args = argparse.Namespace(split="validation", **base)
    val_records, excluded = apex_cli._select_records(val_args)
    assert excluded == set()
    assert len({r.world_id for r in val_records}) == 1

    all_args = argparse.Namespace(split="all", **base)
    all_records, excluded_all = apex_cli._select_records(all_args)
    assert len(all_records) == len(pool)
    assert excluded_all == {val_records[0].world_id}

    capped_args = argparse.Namespace(split="all", **{**base, "test_size": 4})
    capped, _ = apex_cli._select_records(capped_args)
    assert len(capped) == 4
    assert len({r.world_id for r in capped}) == 4


def test_cli_run_contains_per_task_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """One task raising in ``run``'s fan-out records an error row instead of
    aborting the batch."""
    import argparse

    from apex_agents import cli as apex_cli

    records = [
        _eval_record("ib-good", "world-a", rubric=()),
        _eval_record("ib-bad", "world-b", rubric=()),
    ]
    monkeypatch.setattr(apex_cli, "_select_records", lambda _args: (records, set()))

    class _FlakyAgent(ApexReActAgent):
        async def forward(self, *, record: ApexAgentsRecord) -> Any:
            if record.task_id == "ib-bad":
                raise RuntimeError("boom")
            return await super().forward(record=record)

    monkeypatch.setattr(
        apex_cli,
        "_build_agent",
        lambda _args, _config: _FlakyAgent(
            model_name="scripted/test",
            max_steps=5,
            world_factory=lambda _record: FakeWorld({}),
            model_factory=_scripted_model_factory(),
        ),
    )

    args = argparse.Namespace(
        split="all",
        max_concurrency=2,
        output_dir=tmp_path,
        task_model="scripted/test",
        task_temperature=0.0,
        judge_model="stub/judge",
        max_steps=5,
        cost_limit=1.0,
        llm_timeout=5.0,
    )
    assert apex_cli._run_run(args) == 0
    results = json.loads((tmp_path / "run_outputs.json").read_text())
    by_id = {r["task_id"]: r for r in results}
    assert "RuntimeError: boom" in by_id["ib-bad"]["error"]
    assert by_id["ib-good"]["final_answer"] == "The EV is $5M."


def test_cli_run_reports_a_failed_agent_run_as_an_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An agent run that failed without raising is an error row, not a success."""
    import argparse

    from apex_agents import cli as apex_cli

    monkeypatch.setattr(
        apex_cli, "_select_records", lambda _args: ([_eval_record("ib-1", "world-a", rubric=())], set())
    )

    def _exploding_factory(_name: str, _temp: float) -> Any:
        class _M:
            def complete(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
                raise TimeoutError("provider timed out")

        return _M()

    monkeypatch.setattr(
        apex_cli,
        "_build_agent",
        lambda _args, _config: ApexReActAgent(
            model_name="scripted/test",
            max_steps=5,
            world_factory=lambda _record: FakeWorld({}),
            model_factory=_exploding_factory,
        ),
    )

    args = argparse.Namespace(
        split="all",
        max_concurrency=1,
        output_dir=tmp_path,
        task_model="scripted/test",
        task_temperature=0.0,
        judge_model="stub/judge",
        max_steps=5,
        cost_limit=1.0,
        llm_timeout=5.0,
    )
    assert apex_cli._run_run(args) == 0
    (result,) = json.loads((tmp_path / "run_outputs.json").read_text())
    assert "provider timed out" in result["error"]
    assert "final_answer" not in result
