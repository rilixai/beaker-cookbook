"""Consolidated APEX-Agents test suite (non-agent).

The FakeWorld-driven ReAct loop behavioral suite lives in
``test_apex_agents_agent.py``. This file consolidates the rest:

- dataset normalization + IB filter + record_to_case + WorldFiles
- pipeline config + runtime + CLI plumbing
- metrics + LLM judge (incl. parse-verdict calibration)
- world-level k-fold + val splitters
- spec/adapter wiring (end-to-end through stub judge + FakeWorld)
- per-component feedback (Fix-2 CRITICAL non-exploration headline)
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from rilixai.prompt_optimization.evaluation import evaluate_candidate_on_cases
from rilixai.prompt_optimization.models import Case, PromptCandidate
from rilixai.prompt_optimization.spec import build_adapter_from_spec, validate_spec

from apex_agents.agent.types import (
    AgentToolCall,
    ApexAgentsAgentOutput,
)
from apex_agents.data.dataset import (
    _APEX_AGENTS_GROUND_TRUTH_KEY,
    ApexAgentsRecord,
    RubricCriterion,
    cases_from_records,
    filter_investment_banking,
    load_apex_agents_records,
    record_to_case,
    world_ids_for_cases,
)
from apex_agents.data.world_splits import (
    fixed_val_split,
    stratified_case_cap,
    world_held_out_val_split,
    world_level_folds,
)
from apex_agents.feedback import (
    _system_prompt_feedback,
)
from apex_agents.metrics import (
    RUBRIC_FIELD,
    build_rubric_judge,
    score_rubric,
)
from apex_agents.rilixai_spec import ApexAgentsMetrics, ApexAgentsRunner
from apex_agents.tests.fake_world import FakeWorld, fake_world_factory


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


def test_cases_from_records_normalizes_and_exposes_ground_truth() -> None:
    """One broad check that the loader emits Case + ApexAgentsRecord with
    the expected shape AND that ground_truth is bundle-keyed + flat-keyed."""
    cases = cases_from_records([_task_row(0)])
    assert len(cases) == 1
    case = cases[0]
    assert isinstance(case, Case)
    assert case.case_id == "ib-task-0"
    record = case.input
    assert isinstance(record, ApexAgentsRecord)
    assert record.domain == "Investment Banking"
    assert record.world_id == "world-0"
    assert len(record.rubric) == 2
    assert record.rubric[0].verifier_id == "output_llm"
    assert record.task_input_files == ("inputs/brief_0.pdf",)
    # Group key defaults to world_id.
    assert case.group_key == "world-0"
    # Ground-truth bundle exposes task_id/world_id/rubric AND flat rubric.
    bundle = case.ground_truth[_APEX_AGENTS_GROUND_TRUTH_KEY]
    assert isinstance(bundle, Mapping)
    assert bundle["task_id"] == "ib-task-0"
    assert bundle["world_id"] == "world-0"
    assert len(bundle["rubric"]) == 2
    assert case.ground_truth["rubric"][0]["verifier_id"] == "output_llm"


def test_rubric_parses_json_string_and_skips_blank_criteria() -> None:
    row = _task_row(0, rubric=json.dumps([{"verifier_id": "output_llm", "criteria": "X"}]))
    rec = cases_from_records([row])[0].input
    assert len(rec.rubric) == 1
    assert rec.rubric[0].criteria == "X"

    row2 = _task_row(0, rubric=[{"verifier_id": "output_llm"}, {"criteria": "  "}, {"criteria": "real"}])
    rec2 = cases_from_records([row2])[0].input
    assert [c.criteria for c in rec2.rubric] == ["real"]
    assert rec2.rubric[0].verifier_id == "output_llm"


def test_cases_from_records_rejects_unsupported_items() -> None:
    with pytest.raises(TypeError):
        cases_from_records([object()])  # type: ignore[list-item]


def test_record_to_case_explicit_group_key_wins() -> None:
    record = ApexAgentsRecord(
        task_id="t-2",
        task_name="name",
        domain="Investment Banking",
        prompt="...",
        world_id="world-z",
        rubric=(),
        task_input_files=(),
        raw_task={"task_id": "t-2"},
    )
    case = record_to_case(record, group_key="train-split")
    assert case.group_key == "train-split"


def test_filter_ib_case_insensitive_and_world_ids_sorted_distinct() -> None:
    records = [
        ApexAgentsRecord("a", "", "Investment Banking", "p", "w1", (), (), {}),
        ApexAgentsRecord("b", "", "investment banking", "p", "w2", (), (), {}),
        ApexAgentsRecord("c", "", "Law", "p", "w3", (), (), {}),
    ]
    ib = filter_investment_banking(records)
    assert {r.task_id for r in ib} == {"a", "b"}

    cases = cases_from_records([_task_row(i) for i in range(6)])
    assert world_ids_for_cases(cases) == ["world-0", "world-1", "world-2"]


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
    """Fix 3: legal worlds are .docx-heavy; without read_docx those cases
    were unwinnable (read_file returned binary gibberish)."""
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
    """Fix 3: openpyxl is .xlsx-only; legacy .xls must route to xlrd."""
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
    """Fix 3: files named .pdf that are actually ZIP/Office containers must
    return an actionable routing message, not opaque pypdf failure."""
    from apex_agents.agent.world.world import WorldFiles

    (tmp_path / "survey.pdf").write_bytes(b"PK\x03\x04\x14\x00\x00\x00rest-of-zip")
    msg = WorldFiles(tmp_path).read_pdf("survey.pdf")
    assert "NOT a PDF" in msg
    assert "read_docx" in msg and "read_spreadsheet" in msg


# ─────────────────────────────────────────────────────────────────────
# Section 2: pipeline + runtime + CLI plumbing
# ─────────────────────────────────────────────────────────────────────


def _pipeline_record() -> ApexAgentsRecord:
    return ApexAgentsRecord(
        task_id="ib-1",
        task_name="memo",
        domain="Investment Banking",
        prompt="State the EV.",
        world_id="world-a",
        rubric=(RubricCriterion("output_llm", "States an EV."),),
        task_input_files=(),
        raw_task={"task_id": "ib-1"},
    )


def _scripted_model() -> Any:
    class _M:
        def complete(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                "content": "done",
                "tool_calls": [{"id": "c1", "name": "final_answer", "arguments": {"answer": "The EV is $5M."}}],
                "cost": 0.0,
            }

    return _M()


def _offline_apex_spec(
    monkeypatch: Any,
    *,
    judge: Any,
    world_factory: Any = None,
    model_factory: Any = None,
    cases: Any = None,
) -> Any:
    """Build an APEX spec offline through the ``@spec`` path.

    Injects the world factory / model factory / judge the runner would otherwise
    build against the network via ``ctx.metadata``, and bypasses the HF-backed
    ``cases_by_split`` so the test stays hermetic.
    """
    from rilixai.adapters.spec_builder import build_spec_from_runner_class
    from rilixai.testing import stub_optimization_context

    rows = list(cases or [])
    monkeypatch.setattr(ApexAgentsRunner, "cases_by_split", lambda self, ctx: {"train": rows, "validation": rows})
    metadata: dict[str, Any] = {"judge": judge}
    if world_factory is not None:
        metadata["world_factory"] = world_factory
    if model_factory is not None:
        metadata["model_factory"] = model_factory
    ctx = stub_optimization_context(
        config={"domain": "Investment Banking", "train_size": 1, "val_size": 1},
        metadata=metadata,
    )
    return build_spec_from_runner_class(ApexAgentsRunner, ctx)


def test_runner_emits_default_feedback_and_scores_with_stub_judge(monkeypatch: Any) -> None:
    judged: list[tuple[str, str]] = []

    def _stub_judge(criterion: str, answer: str, task_prompt: str) -> bool:
        judged.append((criterion, answer))
        return "EV" in answer

    # Build the spec via the @spec path (injecting offline factories + stub
    # judge through ctx.metadata) and drive its extraction_runtime the way the
    # optimizer adapter would.
    record = _pipeline_record()
    spec = _offline_apex_spec(
        monkeypatch,
        world_factory=lambda _r: FakeWorld({}),
        model_factory=lambda _n, _t: _scripted_model(),
        judge=_stub_judge,
    )
    result = asyncio.run(
        spec.extraction_runtime(
            input=record,
            candidate=spec.seed_candidate,
            case_id=record.task_id,
            ground_truth=record_to_case(record).ground_truth,
        )
    )

    assert result.rubric_pass_rate == 1.0  # forwarded from the _ApexResult output
    assert judged and judged[0][1] == "The EV is $5M."
    feedback = result.run_metrics["trace_evidence"]["per_component_feedback"]
    assert set(feedback) == {"system_prompt", "task_template", "resum_summary_prompt"}
    assert len(set(feedback.values())) == 1  # GenericFeedback uses one shared narrative.
    narrative = feedback["system_prompt"]
    assert "case_id: ib-1" in narrative
    assert "The EV is $5M." in narrative
    assert "expected:" in narrative
    aa = result.run_metrics["apex_agents"]
    assert aa["task_id"] == "ib-1"
    assert aa["world_id"] == "world-a"
    assert aa["rubric_pass_rate"] == 1.0


def test_runner_applies_candidate_prompts_from_runner_state(monkeypatch: Any) -> None:
    seen: list[list[dict[str, Any]]] = []

    class _M:
        def complete(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            seen.append([dict(m) for m in messages])
            return {
                "content": "done",
                "tool_calls": [{"id": "c1", "name": "final_answer", "arguments": {"answer": "The EV is $5M."}}],
                "cost": 0.0,
            }

    spec = _offline_apex_spec(
        monkeypatch,
        world_factory=lambda _r: FakeWorld({}),
        model_factory=lambda _n, _t: _M(),
        judge=lambda *_: True,
    )
    candidate = PromptCandidate(
        components={
            **spec.seed_candidate.components,
            "system_prompt": "RUNNER_STATE_SYSTEM_PROMPT",
            "task_template": "RUNNER_STATE_TASK :: {{task}}",
        }
    )

    asyncio.run(spec.extraction_runtime(input=_pipeline_record(), candidate=candidate))

    assert seen, "scripted model was not called"
    assert "RUNNER_STATE_SYSTEM_PROMPT" in seen[0][0]["content"]
    assert "RUNNER_STATE_TASK" in seen[0][1]["content"]


def test_runner_validates_dict_config_directly() -> None:
    """Direct runner construction should honor mapping-style ctx.config too."""
    from rilixai.testing import stub_optimization_context

    ctx = stub_optimization_context(
        config={
            "domain": "Investment Banking",
            "train_size": 3,
            "val_size": 4,
            "task_model": "test-task-model",
            "task_temperature": 0.2,
            "judge_model": "test-judge-model",
            "max_steps": 5,
            "cost_limit": 1.5,
        },
        metadata={
            "judge": lambda *_: True,
            "world_factory": lambda _record: FakeWorld({}),
        },
    )
    runner = ApexAgentsRunner(ctx)

    assert runner._sandbox_cfg.domain == "Investment Banking"
    assert runner._sandbox_cfg.train_size == 3
    assert runner._sandbox_cfg.val_size == 4
    assert runner.cfg.task_model == "test-task-model"
    assert runner.cfg.task_temperature == pytest.approx(0.2)
    assert runner.cfg.judge_model == "test-judge-model"
    assert runner.cfg.max_steps == 5
    assert runner.cfg.cost_limit == pytest.approx(1.5)


# ─────────────────────────────────────────────────────────────────────
# Section 3: metrics + LLM judge
# ─────────────────────────────────────────────────────────────────────


def _ground_truth(task_id: str, *, n_criteria: int = 2) -> dict[str, Any]:
    rubric = [{"verifier_id": "output_llm", "criteria": f"crit {i} for {task_id}"} for i in range(n_criteria)]
    return {
        "task_id": task_id,
        "world_id": "w",
        "prompt": "p",
        "rubric": rubric,
        _APEX_AGENTS_GROUND_TRUTH_KEY: {
            "task_id": task_id,
            "world_id": "w",
            "prompt": "p",
            "rubric": rubric,
        },
    }


def test_metrics_calculator_aggregates_and_handles_missing_or_empty() -> None:
    metrics = ApexAgentsMetrics()
    assert metrics.field_weights == {RUBRIC_FIELD: 1.0}

    # 3 cases with mixed pass rates → mean.
    out = metrics.calculate_metrics(
        {
            "case-0": {"rubric_pass_rate": 1.0},
            "case-1": {"rubric_pass_rate": 0.5},
            "case-2": {"rubric_pass_rate": 0.0},
        },
        {
            "case-0": _ground_truth("case-0"),
            "case-1": _ground_truth("case-1"),
            "case-2": _ground_truth("case-2"),
        },
    )
    assert out.field_accuracies[RUBRIC_FIELD] == (1.0 + 0.5 + 0.0) / 3
    assert out.field_sample_counts[RUBRIC_FIELD] == 3

    # Cases without rubric criteria are skipped.
    no_rubric = _ground_truth("case-x")
    no_rubric["rubric"] = []
    no_rubric[_APEX_AGENTS_GROUND_TRUTH_KEY]["rubric"] = []
    out2 = metrics.calculate_metrics(
        {"case-x": {"rubric_pass_rate": 1.0}, "case-y": {"rubric_pass_rate": 1.0}},
        {"case-x": no_rubric, "case-y": _ground_truth("case-y")},
    )
    assert out2.field_accuracies[RUBRIC_FIELD] == 1.0
    assert out2.field_sample_counts[RUBRIC_FIELD] == 1

    # Missing result for a case → counts as 0 but the case is counted.
    out3 = metrics.calculate_metrics({}, {"case-0": _ground_truth("case-0")})
    assert out3.field_accuracies[RUBRIC_FIELD] == 0.0
    assert out3.field_sample_counts[RUBRIC_FIELD] == 1

    # Empty {} / {} → 0/0 returns 0.0.
    out4 = metrics.calculate_metrics({}, {})
    assert out4.field_accuracies[RUBRIC_FIELD] == 0.0
    assert out4.field_sample_counts[RUBRIC_FIELD] == 0


def test_comparison_method_clamps_to_unit_interval() -> None:
    metrics = ApexAgentsMetrics()
    comparator = metrics._get_comparison_method(metrics.field_configs[0])
    assert comparator(0.75, _ground_truth("c")) == 0.75
    assert comparator(2.0, _ground_truth("c")) == 1.0
    assert comparator(-1.0, _ground_truth("c")) == 0.0
    assert comparator(None, _ground_truth("c")) == 0.0
    assert comparator(True, _ground_truth("c")) == 1.0


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
    judges (gemini-2.5-flash) vs terse ones (gpt-4.1).
    """
    from apex_agents.metrics import _parse_verdict

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
    # Regression: a last line that merely *starts* with a "MET…" word
    # (METHOD / METADATA / METRICS) must NOT be scored Met — the old
    # startswith("MET") broadening inflated rubric_pass_rate on these.
    assert _parse_verdict("METHODOLOGY looks sound") is False
    assert _parse_verdict("METADATA is complete") is False


# ─────────────────────────────────────────────────────────────────────
# Section 4: k-fold splitters + val splits
# ─────────────────────────────────────────────────────────────────────


def _worlds(n: int) -> list[str]:
    return [f"world-{i:02d}" for i in range(n)]


def test_world_level_folds_shape_partition_and_determinism() -> None:
    worlds = _worlds(10)
    # 10 worlds / k=5 → 5 folds of 2 test / 8 train; disjoint; every world appears once.
    folds = world_level_folds(worlds, k=5, seed=0)
    assert len(folds) == 5
    seen: list[str] = []
    for train, test in folds:
        assert len(test) == 2
        assert len(train) == 8
        assert not (set(train) & set(test))
        seen.extend(test)
    assert sorted(seen) == sorted(worlds)
    assert len(seen) == len(set(seen))

    # Same seed → identical; different seed → at least some fold differs;
    # input order does not matter; dedup of inputs.
    assert world_level_folds(worlds, k=5, seed=7) == world_level_folds(worlds, k=5, seed=7)
    assert [t for _, t in world_level_folds(worlds, k=5, seed=0)] != [
        t for _, t in world_level_folds(worlds, k=5, seed=1)
    ]
    assert world_level_folds(worlds, k=5, seed=3) == world_level_folds(list(reversed(worlds)), k=5, seed=3)
    dedup_folds = world_level_folds(["w1", "w1", "w2", "w3", "w4"], k=2, seed=0)
    deduped_seen: list[str] = []
    for _, test in dedup_folds:
        deduped_seen.extend(test)
    assert sorted(deduped_seen) == ["w1", "w2", "w3", "w4"]


def test_world_level_folds_uneven_balances_and_rejects_bad_args() -> None:
    # 11 worlds / k=5 → sizes differ by ≤ 1, total 11.
    sizes = sorted(len(t) for _, t in world_level_folds(_worlds(11), k=5, seed=0))
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == 11

    with pytest.raises(ValueError, match="k >= 2"):
        world_level_folds(_worlds(10), k=1)
    with pytest.raises(ValueError, match="at least k"):
        world_level_folds(_worlds(3), k=5)


class _C:
    """Minimal Case-like stub: only ``group_key`` matters to the splitter."""

    def __init__(self, world: str, idx: int) -> None:
        self.group_key = world
        self.case_id = f"{world}-{idx}"


def _cases(n_worlds: int, per_world: int = 3) -> list[_C]:
    return [_C(f"world-{w:02d}", i) for w in range(n_worlds) for i in range(per_world)]


def test_inner_val_holds_out_whole_worlds_disjoint_and_deterministic() -> None:
    """Fix 1: cross-world generalization — val split must hold out WHOLE worlds."""
    train = _cases(9, per_world=4)  # 9 worlds, 36 cases
    it, val = world_held_out_val_split(train, n_val_worlds=2, seed=0)
    it_worlds = {c.group_key for c in it}
    val_worlds = {c.group_key for c in val}
    assert it_worlds.isdisjoint(val_worlds)
    assert len(val_worlds) == 2
    assert it_worlds | val_worlds == {f"world-{w:02d}" for w in range(9)}
    assert len(it) + len(val) == len(train)
    it2, val2 = world_held_out_val_split(train, n_val_worlds=2, seed=0)
    assert {c.case_id for c in val2} == {c.case_id for c in val}
    assert {c.group_key for c in world_held_out_val_split(train, n_val_worlds=2, seed=7)[1]} != val_worlds


def test_inner_val_clamps_and_degrades_safely() -> None:
    it, val = world_held_out_val_split(_cases(3), n_val_worlds=10, seed=0)
    assert {c.group_key for c in it} and {c.group_key for c in val}
    assert {c.group_key for c in it}.isdisjoint({c.group_key for c in val})
    one = _cases(1, per_world=5)
    it1, val1 = world_held_out_val_split(one, n_val_worlds=2, seed=0)
    assert len(it1) == len(val1) == 5
    assert world_held_out_val_split([], n_val_worlds=2, seed=0) == ([], [])


def test_fixed_val_split_constant_and_disjoint() -> None:
    cases = _cases(9, per_world=10)
    tp, val, vw = fixed_val_split(cases, n_val_worlds=2, val_size=20, seed=0)
    assert len(vw) == 2
    assert {c.group_key for c in tp}.isdisjoint({c.group_key for c in val})
    assert {c.group_key for c in val} == vw
    assert len(val) == 20
    _, val2, vw2 = fixed_val_split(cases, n_val_worlds=2, val_size=20, seed=0)
    assert [c.case_id for c in val2] == [c.case_id for c in val] and vw2 == vw
    assert len({c.group_key for c in val}) == 2


def test_stratified_cap_keeps_worlds_wide_vs_frontslice() -> None:
    pool = _cases(9, per_world=10)
    strat = stratified_case_cap(pool, 9, mode="stratified", seed=0)
    front = stratified_case_cap(pool, 9, mode="frontslice", seed=0)
    assert len(strat) == 9 and len({c.group_key for c in strat}) == 9
    assert len(front) == 9 and len({c.group_key for c in front}) == 1
    assert stratified_case_cap(pool, None) == pool
    assert [c.case_id for c in stratified_case_cap(pool, 18, seed=0)] == [
        c.case_id for c in stratified_case_cap(pool, 18, seed=0)
    ]
    assert len({c.group_key for c in stratified_case_cap(pool, 18, seed=0)}) == 9


# ─────────────────────────────────────────────────────────────────────
# Section 5: spec / adapter wiring
# ─────────────────────────────────────────────────────────────────────


def _spec_task_row(idx: int) -> dict[str, Any]:
    return {
        "task_id": f"ib-{idx}",
        "task_name": f"memo {idx}",
        "domain": "Investment Banking",
        "prompt": f"State the enterprise value for deal {idx}.",
        "world_id": f"world-{idx}",
        "rubric": [{"verifier_id": "output_llm", "criteria": "Answer mentions an enterprise value."}],
    }


def _scripted_model_factory() -> Any:
    class _M:
        def complete(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                "content": "Submitting the memo.",
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "final_answer",
                        "arguments": {"answer": "The enterprise value is $20M.", "status": "completed"},
                    }
                ],
                "cost": 0.0,
            }

    return lambda _name, _temp: _M()


def _spec_stub_judge(criterion: str, answer: str, task_prompt: str) -> bool:
    return "enterprise value" in answer.lower()


def test_spec_passes_validation_and_carries_spec_metadata(monkeypatch: Any) -> None:
    spec = _offline_apex_spec(monkeypatch, world_factory=fake_world_factory({}), judge=_spec_stub_judge)
    validate_spec(spec)
    assert set(spec.seed_candidate.components.keys()) == {
        "system_prompt",
        "task_template",
        "resum_summary_prompt",
    }
    assert spec.name == "apex-agents"
    assert spec.task_type == "apex_agent"
    assert spec.reflection_evidence_mode == "curated_plus_trace"
    # The default field weights flow through to the profile.
    profile = spec.evaluation_profile_resolver()
    assert profile.field_weights == {"rubric_pass_rate": 1.0}
    # The runtime is a BaseCaseRunner instance with an async __call__.
    assert callable(spec.extraction_runtime)
    assert asyncio.iscoroutinefunction(spec.extraction_runtime.__call__)


def test_spec_end_to_end_via_adapter_with_fake_world_and_stub_judge(monkeypatch: Any) -> None:
    cases = cases_from_records([_spec_task_row(0), _spec_task_row(1)])
    spec = _offline_apex_spec(
        monkeypatch,
        world_factory=fake_world_factory({"brief.txt": "value"}),
        model_factory=_scripted_model_factory(),
        judge=_spec_stub_judge,
        cases=cases,
    )
    validate_spec(spec)
    adapter = build_adapter_from_spec(spec)
    report = evaluate_candidate_on_cases(
        adapter=adapter,
        candidate=spec.seed_candidate,
        cases=cases,
    )
    assert report.field_accuracies["rubric_pass_rate"] == 1.0
    assert report.field_sample_counts["rubric_pass_rate"] == 2
    assert report.weighted_objective == 1.0


# ─────────────────────────────────────────────────────────────────────
# Section 6: per-component feedback (Fix 2)
# ─────────────────────────────────────────────────────────────────────


def _feedback_record() -> ApexAgentsRecord:
    return ApexAgentsRecord(
        task_id="law-1",
        task_name="Indemnity analysis",
        domain="Law",
        prompt="Analyze the indemnity clause.",
        world_id="world-a",
        rubric=(RubricCriterion("output_llm", "States the indemnity cap."),),
        task_input_files=(),
        raw_task={"task_id": "law-1"},
    )


def _asst(idx: int, tool: str | None) -> AgentToolCall:
    return AgentToolCall(step_index=idx, role="assistant", content="...", tool_name=tool)


def test_zero_file_tools_yields_critical_headline_first() -> None:
    out = ApexAgentsAgentOutput(
        final_answer="Based on general principles, the cap is typically 1x fees.",
        status="completed",
        messages=[_asst(0, "todo_write"), _asst(1, "final_answer")],
    )
    fb = _system_prompt_feedback(record=_feedback_record(), output=out)
    assert fb.startswith("CRITICAL FAILURE"), fb[:80]
    assert "ZERO workspace-reading tool calls" in fb
    assert fb.index("CRITICAL FAILURE") < fb.index("You are optimizing")


def test_punt_after_exploring_yields_high_priority_headline() -> None:
    out = ApexAgentsAgentOutput(
        final_answer="I cannot determine the cap without the executed agreement.",
        status="blocked",
        messages=[_asst(0, "list_files"), _asst(1, "read_pdf"), _asst(2, "final_answer")],
    )
    fb = _system_prompt_feedback(record=_feedback_record(), output=out)
    assert fb.startswith("HIGH-PRIORITY FAILURE"), fb[:80]
    assert "PUNTED" in fb
    assert "ZERO workspace-reading" not in fb


def test_explored_and_answered_has_no_failure_headline() -> None:
    out = ApexAgentsAgentOutput(
        final_answer="The indemnity cap is HK$3,125,000 per clause 8.2 of the SPA.",
        status="completed",
        messages=[_asst(0, "list_files"), _asst(1, "read_file"), _asst(2, "final_answer")],
    )
    fb = _system_prompt_feedback(record=_feedback_record(), output=out)
    assert fb.startswith("You are optimizing")
    assert "CRITICAL FAILURE" not in fb and "HIGH-PRIORITY FAILURE" not in fb
