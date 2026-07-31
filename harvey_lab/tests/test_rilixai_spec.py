"""Tests for the RilixAI prompt-optimization spec (``.rilix-ai/rilixai_spec.py``).

These cover the two things that silently break a prompt-optimization
integration: a dataset contract that drifts from the exported rows, and
candidate prompts that never actually reach the agent. The rollout test runs the
**real** ``HarveyLabAgent`` against a fixture task with a scripted Stirrup client
and a stub rubric judge, so it asserts the candidate prompts on the wire without
a single network call.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from rilixai import Case, CaseResult, DatasetRowContext, OptimizationContext, validate_spec

from harvey_lab.agent.agent import HarveyLabAgent
from harvey_lab.agent.prompts import load_harvey_lab_prompts
from harvey_lab.agent.workspace import task_source_from_dir
from harvey_lab.config import HarveyLabConfig
from harvey_lab.evaluation.scoring import ALL_PASS_FIELD, CRITERION_PASS_RATE_FIELD
from tests.test_units import _fee_judge, _local_exec_factory, _ScriptedClient, _write_task


COOKBOOK_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = COOKBOOK_ROOT / ".rilix-ai" / "rilixai_spec.py"
DATASET_DIR = COOKBOOK_ROOT / ".rilix-ai" / "dataset"

SYSTEM_MARKER = "CANDIDATE-SYSTEM-PROMPT-MARKER"
TEMPLATE_MARKER = "CANDIDATE-TASK-TEMPLATE-MARKER"


@pytest.fixture(scope="module")
def spec_module() -> ModuleType:
    """Import the spec by path (``.rilix-ai`` is not an importable package name)."""
    module_spec = importlib.util.spec_from_file_location("rilixai_spec_under_test", SPEC_PATH)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    # dataclasses resolve their annotations through sys.modules, so register the
    # module before executing it.
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


@pytest.fixture
def lab_task_root(tmp_path: Path) -> Path:
    root = tmp_path / "tasks"
    _write_task(
        root,
        "contracts/t1",
        criteria=[
            {
                "id": "C1",
                "title": "States the fee",
                "match_criteria": "Mentions $50,000.",
                "deliverables": ["memo.md"],
            },
            {"id": "C2", "title": "Cites source", "match_criteria": "Cites notes.txt.", "deliverables": ["memo.md"]},
        ],
        deliverables={"memo.md": "Memo"},
    )
    return root


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "contracts/t1",
        "input": {
            "task_id": "contracts/t1",
            "title": "task contracts/t1",
            "work_type": "analyze",
            "instructions": "Summarize the termination fee and cite the source document.",
            "deliverables": {"memo.md": "Memo"},
            "documents": ["notes.txt"],
        },
        "expected": {
            "criteria": [
                {"id": "C1", "title": "States the fee", "match_criteria": "Mentions $50,000.", "deliverables": []},
            ]
        },
        "metadata": {"practice_area": "contracts"},
    }
    row.update(overrides)
    return row


# ─── seed targets ─────────────────────────────────────────────────────


def test_seed_targets_are_the_prompts_the_recipe_ships(spec_module: ModuleType) -> None:
    system_prompt, task_template = load_harvey_lab_prompts()
    assert spec_module._seed_targets().to_dict() == {
        "system_prompt": system_prompt,
        "task_template": task_template,
    }


# ─── dataset contract ─────────────────────────────────────────────────


def test_loader_builds_cases_from_a_row(spec_module: ModuleType) -> None:
    loader = spec_module.LabTaskDataLoader()
    context = DatasetRowContext(split="train", line_number=1)
    cases = list(loader.iter_cases(loader.parse_row(_row(), context), context))
    assert len(cases) == 1
    case = cases[0]
    assert case.case_id == "contracts/t1"
    assert case.group_key == "contracts"
    assert case.input["instructions"].startswith("Summarize")
    assert [criterion["id"] for criterion in case.ground_truth["criteria"]] == ["C1"]


@pytest.mark.parametrize(
    "row",
    [
        _row(expected={"criteria": []}),
        _row(expected={"criteria": [{"id": "C1", "title": "no standard"}]}),
        _row(input={"task_id": "contracts/t1", "instructions": "  "}),
        _row(id="", input={"instructions": "Summarize the fee."}),
        _row(expected=["not", "an", "object"]),
    ],
)
def test_loader_rejects_unusable_rows(spec_module: ModuleType, row: dict[str, Any]) -> None:
    loader = spec_module.LabTaskDataLoader()
    with pytest.raises((ValueError, TypeError)):
        loader.parse_row(row, DatasetRowContext(split="train", line_number=1))


@pytest.mark.skipif(not DATASET_DIR.exists(), reason="local dataset export not present")
def test_exported_dataset_matches_the_declared_contract(spec_module: ModuleType) -> None:
    loader = spec_module.LabTaskDataLoader()
    for split in ("train", "val"):
        path = DATASET_DIR / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert rows, f"{path} is empty"
        for index, raw in enumerate(rows, start=1):
            context = DatasetRowContext(split=split, path=str(path), line_number=index)
            cases = list(loader.iter_cases(loader.parse_row(raw, context), context))
            assert len(cases) == 1
            assert cases[0].ground_truth["criteria"], f"{context.label()}: unscoreable row"


# ─── rollout ──────────────────────────────────────────────────────────


def _patch_for_local_rollout(
    spec_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    lab_task_root: Path,
    *,
    body: str,
) -> list[list[Any]]:
    """Wire ``_run_case`` to a fixture task, a scripted client and a stub judge.

    Everything under test stays real: the actual ``HarveyLabAgent``, its prompt
    rendering, its ``code_exec`` workspace and the recipe's rubric aggregation.
    Returns the list the client records each turn's messages into.
    """
    seen_messages: list[list[Any]] = []

    class _RecordingClient(_ScriptedClient):
        async def generate(self, messages: list[Any], tools: dict[str, Any]) -> Any:
            seen_messages.append(list(messages))
            return await super().generate(messages, tools)

    def _model_factory(*_args: Any) -> Any:
        return _RecordingClient(deliverable_body=body, deliverable_name="memo.md")

    record = spec_module.load_records(lab_task_root, task_ids=["contracts/t1"])[0]
    monkeypatch.setattr(spec_module, "_record_for_case", lambda case, **_kw: (record, lab_task_root))
    monkeypatch.setattr(
        spec_module,
        "_config_for_runtime",
        lambda _runtime: HarveyLabConfig(max_turns=5, enable_view_image=False),
    )
    monkeypatch.setattr(
        spec_module,
        "HarveyLabAgent",
        partial(HarveyLabAgent, model_factory=_model_factory, exec_provider_factory=_local_exec_factory),
    )
    monkeypatch.setattr(spec_module, "build_rubric_judge", lambda *_a, **_kw: _fee_judge)
    return seen_messages


def _candidate_prompts(spec_module: ModuleType) -> Any:
    system_prompt, task_template = load_harvey_lab_prompts()
    return spec_module.optimization_targets_from_prompts(
        {
            "system_prompt": f"{system_prompt}\n\n{SYSTEM_MARKER}",
            "task_template": f"{task_template}\n\n{TEMPLATE_MARKER}",
        }
    )


def test_run_case_sends_the_candidate_prompts_to_the_agent_and_scores_the_rubric(
    spec_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    lab_task_root: Path,
) -> None:
    seen_messages = _patch_for_local_rollout(
        spec_module,
        monkeypatch,
        lab_task_root,
        body="The termination fee is $50,000 per notes.txt.",
    )
    case = Case(
        input={"task_id": "contracts/t1", "title": "task contracts/t1", "instructions": "Summarize the fee."},
        case_id="contracts/t1",
        ground_truth={
            "criteria": [
                {"id": "C1", "title": "States the fee", "match_criteria": "Mentions $50,000.", "deliverables": []},
                {"id": "C2", "title": "Cites source", "match_criteria": "Cites notes.txt.", "deliverables": []},
            ]
        },
    )

    result = asyncio.run(
        spec_module._run_case(case=case, targets=_candidate_prompts(spec_module), runtime=None),
    )
    score = asyncio.run(spec_module.RubricScorer().score_case(case=case, result=result))

    # The candidate prompts reached the model, not the seeds.
    prompt_text = "\n".join(str(getattr(message, "content", "")) for turn in seen_messages for message in turn)
    assert SYSTEM_MARKER in prompt_text
    assert TEMPLATE_MARKER in prompt_text

    assert result.output["deliverables_produced"] == ["memo.md"]
    assert result.output["finished"] is True
    assert score.field_scores == {ALL_PASS_FIELD: 1.0, CRITERION_PASS_RATE_FIELD: 1.0}
    assert score.objective == 1.0
    trace = result.run_metrics["trace_evidence"]
    assert set(trace["per_prompt_feedback"]) == {"system_prompt", "task_template"}


def test_run_case_reports_missing_deliverables_and_failed_criteria(
    spec_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    lab_task_root: Path,
) -> None:
    _patch_for_local_rollout(spec_module, monkeypatch, lab_task_root, body="No numbers here.")
    case = Case(
        input={"task_id": "contracts/t1", "instructions": "Summarize the fee."},
        case_id="contracts/t1",
        ground_truth={
            "criteria": [{"id": "C1", "title": "States the fee", "match_criteria": "Mentions $50,000."}],
        },
    )

    result = asyncio.run(spec_module._run_case(case=case, targets=_candidate_prompts(spec_module), runtime=None))
    score = asyncio.run(spec_module.RubricScorer().score_case(case=case, result=result))

    assert score.objective == 0.0
    assert score.field_scores[ALL_PASS_FIELD] == 0.0
    feedback = result.run_metrics["trace_evidence"]["per_prompt_feedback"]["task_template"]
    assert any("failed criteria" in note for note in feedback)


def test_run_case_failure_scores_zero_instead_of_aborting_the_run(
    spec_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(_case: Case, **_kwargs: Any) -> Any:
        raise RuntimeError("task fetch failed")

    monkeypatch.setattr(spec_module, "_record_for_case", _boom)
    case = Case(
        input={"task_id": "contracts/t1", "instructions": "Summarize the fee."},
        case_id="contracts/t1",
        ground_truth={"criteria": [{"id": "C1", "match_criteria": "Mentions $50,000."}]},
    )

    result = asyncio.run(spec_module._run_case(case=case, targets=_candidate_prompts(spec_module), runtime=None))
    score = asyncio.run(spec_module.RubricScorer().score_case(case=case, result=result))

    assert "task fetch failed" in result.output["error"]
    assert score.field_scores == {ALL_PASS_FIELD: 0.0, CRITERION_PASS_RATE_FIELD: 0.0}


def test_targets_that_miss_the_agent_are_rejected(spec_module: ModuleType, lab_task_root: Path) -> None:
    """The guard that keeps the optimizer from tuning prompts the agent ignores."""
    system_prompt, task_template = load_harvey_lab_prompts()
    agent = HarveyLabAgent(
        config=HarveyLabConfig(max_turns=5, enable_view_image=False),
        task_source=task_source_from_dir(lab_task_root),
        system_prompt=system_prompt,
        task_template=task_template,
    )
    spec_module._assert_targets_applied(agent, {"system_prompt": system_prompt, "task_template": task_template})
    with pytest.raises(RuntimeError, match="task_template"):
        spec_module._assert_targets_applied(agent, {"system_prompt": system_prompt, "task_template": "something else"})
    with pytest.raises(RuntimeError, match="system_prompt"):
        spec_module._assert_targets_applied(
            agent, {"system_prompt": "another system prompt", "task_template": task_template}
        )


def test_record_for_case_rejects_a_task_tree_that_drifted_from_the_row(
    spec_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    lab_task_root: Path,
) -> None:
    monkeypatch.setattr(spec_module, "ensure_task_dirs", lambda *_a, **_kw: lab_task_root)
    case = Case(
        input={"task_id": "contracts/t1", "instructions": "Summarize the fee."},
        case_id="contracts/t1",
        ground_truth={"criteria": []},
        metadata={"task_fingerprint": "deadbeef" * 8},
    )
    with pytest.raises(RuntimeError, match="does not match the dataset row"):
        spec_module._record_for_case(case)


# ─── scoring ──────────────────────────────────────────────────────────


def test_scorer_counts_only_verdicts_for_the_rows_own_criteria(spec_module: ModuleType) -> None:
    case = Case(
        input={},
        case_id="c",
        ground_truth={
            "criteria": [
                {"id": "C1", "match_criteria": "a"},
                {"id": "C2", "match_criteria": "b"},
                {"id": "C3", "match_criteria": "c"},
                {"id": "C4", "match_criteria": "d"},
            ]
        },
    )
    result = CaseResult(
        output={},
        context={
            "verdicts": [
                {"id": "C1", "passed": True},
                {"id": "C2", "passed": False},
                {"id": "C3", "passed": True},
                # A verdict for a criterion this row does not carry is ignored.
                {"id": "C9", "passed": True},
            ]
        },
    )

    score = asyncio.run(spec_module.RubricScorer().score_case(case=case, result=result))

    assert score.field_scores == {ALL_PASS_FIELD: 0.0, CRITERION_PASS_RATE_FIELD: 0.5}
    assert score.objective == 0.5


# ─── spec assembly ────────────────────────────────────────────────────


def test_build_spec_is_valid_and_declares_its_dataset_schema(spec_module: ModuleType) -> None:
    built = validate_spec(spec_module.build_spec(OptimizationContext()))
    assert built.name == "harvey-lab"
    assert built.data_loader.dataset_schema is spec_module.DATASET_SCHEMA
    registration = spec_module.build_spec.__rilixai_spec__
    assert registration.name == "harvey-lab"
    assert registration.metadata["dataset_schema"]["json_schema"]["required"] == ["id", "input", "expected"]
