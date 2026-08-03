"""Contract tests for the Harvey LAB RilixAI optimization spec."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from rilixai import Case, CaseResult, DatasetRowContext, InferenceTarget, OptimizationTargets, RolloutContext

from harvey_lab.agent.agent import HarveyLabAgentOutput
from harvey_lab.config import HARVEY_LABS_COMMIT
from harvey_lab.data.dataset import load_records


BEAKER_DIR = Path(__file__).resolve().parents[1] / ".beaker"
sys.path.insert(0, str(BEAKER_DIR))
import rilixai_spec as spec_mod  # noqa: E402


def _record(tmp_path: Path) -> tuple[Path, Any]:
    tasks_root = tmp_path / "tasks"
    task_dir = tasks_root / "contracts" / "real-task"
    (task_dir / "documents").mkdir(parents=True)
    (task_dir / "documents" / "agreement.txt").write_text("Termination fee: $50,000.")
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "title": "Review termination fee",
                "work_type": "analyze",
                "tags": ["contracts"],
                "instructions": "Prepare a memo identifying the termination fee.",
                "deliverables": {"memo.md": "Memo"},
                "criteria": [
                    {
                        "id": "C1",
                        "title": "States the fee",
                        "match_criteria": "The memo states that the termination fee is $50,000.",
                        "deliverables": ["memo.md"],
                    }
                ],
            }
        )
    )
    return tasks_root, load_records(tasks_root, task_ids=["contracts/real-task"])[0]


def _case() -> Case:
    return Case(
        input={"task_id": "contracts/real-task"},
        case_id="contracts/real-task",
        ground_truth={
            "criteria": [
                {
                    "id": "C1",
                    "title": "States fee",
                    "match_criteria": "Mentions $50,000.",
                    "deliverables": ["memo.md"],
                },
                {
                    "id": "C2",
                    "title": "Cites source",
                    "match_criteria": "Cites agreement.txt.",
                    "deliverables": ["memo.md"],
                },
            ]
        },
        group_key="contracts",
        metadata={"title": "Review", "instructions": "Prepare a memo."},
    )


def test_loader_resolves_real_record_and_rubric(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tasks_root, record = _record(tmp_path)
    monkeypatch.setattr(spec_mod, "_load_record", lambda _task_id: (tasks_root, record))
    loader = spec_mod.HarveyLabDataLoader()
    context = DatasetRowContext(split="train", path="train.jsonl", line_number=1)
    row = loader.parse_row(
        {"task_id": record.task_id, "source_commit": HARVEY_LABS_COMMIT},
        context,
    )
    case = list(loader.iter_cases(row, context))[0]

    assert case.case_id == record.task_id
    assert case.group_key == "contracts"
    assert case.ground_truth["criteria"][0]["match_criteria"].startswith("The memo states")
    assert case.metadata["deliverables"] == {"memo.md": "Memo"}


def test_loader_rejects_dataset_revision_drift() -> None:
    loader = spec_mod.HarveyLabDataLoader()
    with pytest.raises(ValueError, match="pinned Harvey LAB revision"):
        loader.parse_row(
            {"task_id": "contracts/real-task", "source_commit": "wrong"},
            DatasetRowContext(split="train"),
        )


def test_loader_rejects_absolute_task_path() -> None:
    loader = spec_mod.HarveyLabDataLoader()
    with pytest.raises(ValueError, match="relative Harvey LAB task path"):
        loader.parse_row(
            {"task_id": "/contracts/real-task", "source_commit": HARVEY_LABS_COMMIT},
            DatasetRowContext(split="train"),
        )


def test_run_case_applies_both_candidate_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tasks_root, record = _record(tmp_path)
    captured: dict[str, Any] = {}

    class _FakeAgent:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        async def forward(self, *, record: Any) -> HarveyLabAgentOutput:
            captured["record"] = record
            return HarveyLabAgentOutput(
                final_answer="done",
                deliverables={"memo.md": "The fee is $50,000."},
                finished=True,
            )

    monkeypatch.setattr(spec_mod, "_load_record", lambda _task_id: (tasks_root, record))
    monkeypatch.setattr(spec_mod, "HarveyLabAgent", _FakeAgent)
    targets = OptimizationTargets(
        prompts={
            spec_mod.SYSTEM_PROMPT_TARGET: "candidate system",
            spec_mod.TASK_TEMPLATE_TARGET: "candidate task {{instructions}}",
        }
    )
    result = asyncio.run(
        spec_mod._run_case(case=_case(), targets=targets, runtime=RolloutContext(model=None, user_id="test"))
    )

    assert result.failure is None
    assert captured["system_prompt"] == "candidate system"
    assert captured["task_template"] == "candidate task {{instructions}}"
    assert captured["model_factory"] is None
    assert result.output["deliverables"] == {"memo.md": "The fee is $50,000."}


def test_run_case_uses_gateway_factory_only_for_selected_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks_root, record = _record(tmp_path)
    gateway_factory = object()
    captured: dict[str, Any] = {}

    class _FakeAgent:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        async def forward(self, *, record: Any) -> HarveyLabAgentOutput:
            return HarveyLabAgentOutput(final_answer="done", abandoned=True)

    monkeypatch.setattr(spec_mod, "_load_record", lambda _task_id: (tasks_root, record))
    monkeypatch.setattr(spec_mod, "HarveyLabAgent", _FakeAgent)
    monkeypatch.setattr(spec_mod, "_gateway_model_factory", lambda _runtime: gateway_factory)
    runtime = RolloutContext(
        model="gpt-5",
        provider="openai",
        canonical_model_id="openai:gpt-5",
        user_id="test",
    )
    result = asyncio.run(spec_mod._run_case(case=_case(), targets=spec_mod._seed_targets(), runtime=runtime))

    assert result.failure is None
    assert captured["model_factory"] is gateway_factory


def test_gateway_factory_adapts_canonical_model_to_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        spec_mod,
        "inference_target",
        lambda _runtime: InferenceTarget(
            base_url="https://gateway.example/v1/llm",
            api_key="run-scoped-token",
            model="anthropic:claude-opus-4-1",
        ),
    )
    runtime = RolloutContext(
        model="claude-opus-4-1",
        provider="anthropic",
        canonical_model_id="anthropic:claude-opus-4-1",
        user_id="test",
    )
    factory = spec_mod._gateway_model_factory(runtime)
    client = factory("production/model", 0.2, 4096, 30.0, "none")

    assert client.model_slug == "openai/anthropic:claude-opus-4-1"
    assert client._api_key == "run-scoped-token"
    assert client._kwargs["api_base"] == "https://gateway.example/v1/llm"


def test_rubric_scorer_optimizes_criterion_pass_rate() -> None:
    def _judge(_task: str, criteria: list[dict[str, Any]], _output: str) -> dict[str, bool]:
        return {str(criterion["id"]): criterion["id"] == "C1" for criterion in criteria}

    scorer = spec_mod.HarveyLabRubricScorer(judge=_judge)
    result = CaseResult(output={"deliverables": {"memo.md": "The fee is $50,000."}})
    score = asyncio.run(scorer.score_case(case=_case(), result=result))

    assert score.field_scores == {"all_pass": 0.0, "criterion_pass_rate": 0.5}
    assert score.objective == 0.5


def test_seed_targets_match_application_prompts() -> None:
    prompts = spec_mod._seed_targets().to_dict()
    assert prompts[spec_mod.SYSTEM_PROMPT_TARGET] == spec_mod.SYSTEM_PROMPT_SEED
    assert prompts[spec_mod.TASK_TEMPLATE_TARGET] == spec_mod.TASK_TEMPLATE_SEED
    assert "TODO(rilixai)" not in Path(spec_mod.__file__).read_text()
