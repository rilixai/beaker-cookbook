"""Beaker repository optimization spec for AutomationBench filesystem skills."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cache
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
    Spec,
    inference_target,
    spec,
)

from automationbench_skills import ModelSpec, run_one_async
from automationbench_skills.data.tasks import Sample, load_samples


FIELD_NAME = "partial_credit"
_GATEWAY_KEY_VAR = "BEAKER_INFERENCE_API_KEY"
_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


@dataclass(frozen=True)
class _TaskRow:
    id: str
    task_name: str
    expected: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    group_key: str = "default"


class _TaskDataLoader(CaseDataLoader[_TaskRow]):
    dataset_schema = STANDARD_JSONL_CASE_SCHEMA

    def parse_row(self, raw: Mapping[str, Any], context: DatasetRowContext) -> _TaskRow:
        del context
        missing = [name for name in ("id", "input", "expected") if name not in raw]
        if missing:
            raise ValueError(f"missing required field(s): {', '.join(missing)}")
        row_id = str(raw["id"]).strip()
        input_payload = raw["input"]
        expected = raw["expected"]
        metadata = raw.get("metadata") or {}
        if not row_id:
            raise ValueError("id must be non-empty")
        if not isinstance(input_payload, Mapping) or not str(input_payload.get("task_name") or "").strip():
            raise TypeError("input.task_name must be a non-empty string")
        if not isinstance(expected, Mapping) or expected.get(FIELD_NAME) != 1.0:
            raise ValueError("expected.partial_credit must be 1.0")
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a JSON object")
        return _TaskRow(
            id=row_id,
            task_name=str(input_payload["task_name"]),
            expected=dict(expected),
            metadata=dict(metadata),
            group_key=str(raw.get("group_key") or "default"),
        )

    def iter_cases(self, row: _TaskRow, context: DatasetRowContext) -> Iterable[Case]:
        del context
        yield Case(
            input={"task_name": row.task_name},
            case_id=row.id,
            ground_truth=row.expected,
            group_key=row.group_key,
            metadata=row.metadata,
        )


@cache
def _samples_by_name() -> dict[str, Sample]:
    return {sample.task_name: sample for sample in load_samples()}


def _model_for_runtime(runtime: Any) -> ModelSpec:
    if not runtime.model:
        return ModelSpec()
    target = inference_target(runtime)
    os.environ[_GATEWAY_KEY_VAR] = target.api_key
    reasoning_effort = runtime.model_capabilities.reasoning_effort if runtime.model_capabilities else None
    return ModelSpec(
        name=target.model,
        base_url=target.base_url,
        api_key_var=_GATEWAY_KEY_VAR,
        api="chat_completions",
        reasoning_effort=reasoning_effort,
    )


async def _run_case(*, case: Case, targets: None, runtime: Any) -> CaseResult:
    del targets
    task_name = str(case.input.get("task_name") or "")
    sample = _samples_by_name().get(task_name)
    if sample is None:
        return CaseResult.failed(f"Unknown AutomationBench task: {task_name}")
    model = _model_for_runtime(runtime)
    with runtime.trace.stage("automationbench.run", inputs={"task_name": task_name}) as stage:
        result = await run_one_async(sample, model=model, skills_dir=_SKILLS_DIR)
        output = {
            "partial_credit": result.partial_credit,
            "task_completed_correctly": result.task_completed_correctly,
        }
        stage.output(output)
    context = {
        "task_name": result.task_name,
        "domain": result.domain,
        "trajectory": result.trajectory,
        "end_state": result.end_state,
        "assertion_results": result.assertion_results,
    }
    if result.error is not None:
        return CaseResult.failed(
            str(result.error),
            context=context,
            retryable="timeout" in str(result.error).lower(),
        )
    return CaseResult(output=output, context=context)


class _PartialCreditScorer:
    async def score_case(self, *, case: Case, result: CaseResult) -> CaseScore:
        del case
        output = result.output if isinstance(result.output, Mapping) else {}
        value = output.get(FIELD_NAME)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError("prediction.partial_credit must be numeric")
        score = max(0.0, min(1.0, float(value)))
        return CaseScore(field_scores={FIELD_NAME: score}, objective=score, key="automationbench")


@spec(
    dataset_schema=STANDARD_JSONL_CASE_SCHEMA,
    repository=("skills",),
)
def build_spec(ctx: OptimizationContext) -> Spec:
    del ctx
    return Spec(
        data_loader=_TaskDataLoader(),
        run_case=_run_case,
        scorer=_PartialCreditScorer(),
    )
