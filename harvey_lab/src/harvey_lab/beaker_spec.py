"""Beaker prompt-optimization spec for the Harvey LAB legal agent.

The dataset rows reference immutable tasks from the public HARVEY-LABS
benchmark.  Their rubric is the ground truth: the evaluator asks the existing
rubric judge whether the deliverables satisfy each criterion.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from beaker import (
    STANDARD_JSONL_CASE_SCHEMA,
    Case,
    CaseDataLoader,
    CaseResult,
    CaseScore,
    DatasetRowContext,
    OptimizationContext,
    OptimizationTargets,
    RolloutContext,
    Spec,
    inference_target,
    objective_score,
    optimization_targets_from_prompts,
    spec,
)

from .agent.agent import HarveyLabAgent
from .agent.prompts import SYSTEM_PROMPT_SEED, TASK_TEMPLATE_SEED
from .agent.workspace import task_source_from_dir
from .config import HarveyLabConfig
from .data.dataset import HarveyLabRecord, RubricCriterion, load_records
from .data.fetch import ensure_task_dirs
from .evaluation.scoring import (
    ALL_PASS_FIELD,
    CRITERION_PASS_RATE_FIELD,
    JudgeCallError,
    build_rubric_judge,
    score_rubric,
)


PROMPT_TARGETS = ("system_prompt", "task_template")
_EXPECTED_FIELDS = ("task_fingerprint", "title", "instructions", "deliverables", "criteria")


@dataclass(frozen=True)
class HarveyLabDatasetRow:
    """One real Harvey LAB task plus its source rubric ground truth."""

    id: str
    input: dict[str, Any]
    expected: dict[str, Any]
    metadata: dict[str, Any]
    group_key: str


class HarveyLabDataLoader(CaseDataLoader[HarveyLabDatasetRow]):
    """Validate exported HARVEY-LABS JSONL rows and yield one task per case."""

    dataset_schema = STANDARD_JSONL_CASE_SCHEMA

    def parse_row(self, raw: Mapping[str, Any], context: DatasetRowContext) -> HarveyLabDatasetRow:
        missing = [field for field in ("id", "input", "expected") if field not in raw]
        if missing:
            raise ValueError(f"missing required field(s): {', '.join(missing)}")
        row_id = str(raw["id"]).strip()
        if not row_id:
            raise ValueError("id must be non-empty")
        input_payload = raw["input"]
        expected = raw["expected"]
        metadata = raw.get("metadata") or {}
        if not isinstance(input_payload, Mapping):
            raise TypeError("input must be a JSON object")
        if not isinstance(expected, Mapping):
            raise TypeError("expected must be a JSON object")
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a JSON object")
        if str(input_payload.get("task_id") or "").strip() != row_id:
            raise ValueError("input.task_id must match id")
        missing_expected = [field for field in _EXPECTED_FIELDS if field not in expected]
        if missing_expected:
            raise ValueError(f"expected is missing required field(s): {', '.join(missing_expected)}")
        if not isinstance(expected["deliverables"], Mapping):
            raise TypeError("expected.deliverables must be a JSON object")
        if not isinstance(expected["criteria"], Sequence) or isinstance(expected["criteria"], (str, bytes)):
            raise TypeError("expected.criteria must be a JSON array")
        return HarveyLabDatasetRow(
            id=row_id,
            input=dict(input_payload),
            expected=dict(expected),
            metadata=dict(metadata),
            group_key=str(raw.get("group_key") or row_id.split("/", 1)[0] or "default"),
        )

    def iter_cases(self, row: HarveyLabDatasetRow, context: DatasetRowContext) -> Iterable[Case]:
        del context
        yield Case(
            case_id=row.id,
            input=row.input,
            ground_truth=row.expected,
            metadata=row.metadata,
            group_key=row.group_key,
        )


def _seed_targets() -> OptimizationTargets:
    """Return the exact system and per-task prompts used by the legal agent."""
    return optimization_targets_from_prompts(
        {
            "system_prompt": SYSTEM_PROMPT_SEED,
            "task_template": TASK_TEMPLATE_SEED,
        }
    )


def _criteria_from_expected(expected: Mapping[str, Any]) -> tuple[RubricCriterion, ...]:
    criteria: list[RubricCriterion] = []
    for index, value in enumerate(expected["criteria"], start=1):
        if not isinstance(value, Mapping):
            raise ValueError(f"expected.criteria[{index}] must be a JSON object")
        criterion_id = str(value.get("id") or "").strip()
        standard = str(value.get("match_criteria") or "").strip()
        if not criterion_id or not standard:
            raise ValueError(f"expected.criteria[{index}] needs non-empty id and match_criteria")
        deliverables = value.get("deliverables") or []
        if not isinstance(deliverables, Sequence) or isinstance(deliverables, (str, bytes)):
            raise ValueError(f"expected.criteria[{index}].deliverables must be a JSON array")
        criteria.append(
            RubricCriterion(
                id=criterion_id,
                title=str(value.get("title") or ""),
                match_criteria=standard,
                deliverables=tuple(str(name) for name in deliverables),
            )
        )
    return tuple(criteria)


def _record_for_case(case: Case) -> tuple[Path, HarveyLabRecord]:
    task_id = str(case.input.get("task_id") or "").strip() if isinstance(case.input, Mapping) else ""
    if task_id != case.case_id:
        raise ValueError("case.input.task_id must match case_id")
    configured_root = case.input.get("tasks_root") if isinstance(case.input, Mapping) else None
    tasks_root = Path(str(configured_root)) if configured_root else ensure_task_dirs([task_id])
    records = load_records(tasks_root, task_ids=[task_id])
    record = records[0]
    expected = case.ground_truth
    if record.task_fingerprint != str(expected["task_fingerprint"]):
        raise ValueError("downloaded task fingerprint does not match the labeled dataset row")
    if record.title != str(expected["title"]) or record.instructions != str(expected["instructions"]):
        raise ValueError("downloaded task metadata does not match the labeled dataset row")
    if dict(record.deliverables) != {str(name): str(label) for name, label in expected["deliverables"].items()}:
        raise ValueError("downloaded task deliverables do not match the labeled dataset row")
    return tasks_root, replace(record, criteria=_criteria_from_expected(expected))


def _selected_model_factory(runtime: RolloutContext) -> Any:
    """Build a Stirrup client for Beaker's OpenAI-compatible inference proxy."""
    target = inference_target(runtime)

    def factory(_model: str, temperature: float, max_tokens: int, timeout: float, _reasoning_effort: str) -> Any:
        from stirrup.clients.litellm_client import LiteLLMClient

        # LiteLLM strips its ``openai/`` routing prefix before sending the model
        # name.  Keep Beaker's canonical provider:model identifier intact for
        # its gateway rather than attempting to infer a provider-specific route.
        return LiteLLMClient(
            model=f"openai/{target.model}",
            max_tokens=max_tokens,
            api_key=target.api_key,
            kwargs={"api_base": target.base_url, "temperature": temperature, "timeout": timeout},
        )

    return factory


def _agent_for_case(
    *,
    tasks_root: Path,
    targets: OptimizationTargets,
    runtime: RolloutContext,
) -> HarveyLabAgent:
    prompts = targets.to_dict()
    missing = [name for name in PROMPT_TARGETS if not prompts.get(name)]
    if missing:
        raise ValueError(f"optimized prompt targets are missing: {', '.join(missing)}")
    config = HarveyLabConfig()
    model_factory = None
    if runtime.model:
        # A Beaker-selected model is evaluation-only.  Ordinary dry-runs and
        # normal CLI use retain the app's configured OpenRouter defaults.
        config = replace(config, task_reasoning_effort="none")
        model_factory = _selected_model_factory(runtime)
    return HarveyLabAgent(
        config=config,
        task_source=task_source_from_dir(tasks_root),
        model_factory=model_factory,
        system_prompt=prompts["system_prompt"],
        task_template=prompts["task_template"],
    )


async def _run_case(*, case: Case, targets: OptimizationTargets, runtime: RolloutContext) -> CaseResult:
    """Run the real agent with every optimized prompt and retain its deliverables."""
    try:
        with runtime.trace.stage("harvey_lab.load_case", inputs={"task_id": case.case_id}) as stage:
            tasks_root, record = await asyncio.to_thread(_record_for_case, case)
            stage.output({"task_id": record.task_id, "criteria": len(record.criteria)})
        with runtime.trace.stage("harvey_lab.run_agent", inputs={"task_id": case.case_id}) as stage:
            agent = _agent_for_case(tasks_root=tasks_root, targets=targets, runtime=runtime)
            result = await agent.forward(record=record)
            output = {
                "deliverables": result.deliverables,
                "finished": result.finished,
                "abandoned": result.abandoned,
                "max_turns_reached": result.max_turns_reached,
            }
            stage.output({"deliverables": sorted(result.deliverables), "finished": result.finished})
        return CaseResult(output=output, context={"task_id": case.case_id, "total_turns": result.total_turns})
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        return CaseResult.failed(f"Harvey LAB rollout could not run: {type(exc).__name__}: {exc}", retryable=True)


class HarveyLabRubricScorer:
    """Score deliverables against the source task's labeled criterion rubric."""

    async def score_case(self, *, case: Case, result: CaseResult) -> CaseScore:
        output = result.output if isinstance(result.output, Mapping) else {}
        deliverables = output.get("deliverables") if isinstance(output.get("deliverables"), Mapping) else {}
        expected = case.ground_truth
        criteria = [
            {
                "id": criterion.id,
                "title": criterion.title,
                "match_criteria": criterion.match_criteria,
                "deliverables": list(criterion.deliverables),
            }
            for criterion in _criteria_from_expected(expected)
        ]
        try:
            scored = await asyncio.to_thread(
                score_rubric,
                criteria=criteria,
                deliverables={str(name): str(text) for name, text in deliverables.items()},
                task_description=f"{expected['title']}\n\n{expected['instructions']}",
                judge=build_rubric_judge(),
            )
        except JudgeCallError as exc:
            raise RuntimeError(f"Harvey LAB rubric judge could not score the case: {exc}") from exc
        field_scores = {
            ALL_PASS_FIELD: float(scored[ALL_PASS_FIELD]),
            CRITERION_PASS_RATE_FIELD: float(scored[CRITERION_PASS_RATE_FIELD]),
        }
        return CaseScore(
            field_scores=field_scores,
            objective=objective_score(field_scores, field_weights={CRITERION_PASS_RATE_FIELD: 1.0}),
            key="harvey_lab_rubric",
        )


@spec(
    name="harvey-lab-prompts",
    version="v1",
    description="Optimize the system and task prompts used by the Harvey LAB legal-work agent.",
    metadata={"task_type": "agent"},
    dataset_schema=STANDARD_JSONL_CASE_SCHEMA,
)
def build_spec(ctx: OptimizationContext) -> Spec:
    """Build the prompt optimization contract for Harvey LAB tasks."""
    del ctx
    return Spec(
        name="harvey-lab-prompts",
        seed_targets=_seed_targets(),
        data_loader=HarveyLabDataLoader(),
        run_case=_run_case,
        scorer=HarveyLabRubricScorer(),
    )


__all__ = ["HarveyLabDataLoader", "HarveyLabRubricScorer", "build_spec"]
