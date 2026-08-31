"""Beaker repository optimization spec for AutomationBench skills."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
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


METRIC_FIELDS = ("partial_credit", "task_completed_correctly")
OBJECTIVE_FIELD = "partial_credit"
GATEWAY_API_KEY_VAR = "AUTOMATIONBENCH_BEAKER_GATEWAY_KEY"


@dataclass(frozen=True)
class _AutomationBenchRow:
    id: str
    input: dict[str, Any]
    expected: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    group_key: str = "default"


class _AutomationBenchDataLoader(CaseDataLoader[_AutomationBenchRow]):
    """Load task IDs and success targets derived from the frozen benchmark."""

    dataset_schema = STANDARD_JSONL_CASE_SCHEMA

    def parse_row(self, raw: Mapping[str, Any], context: DatasetRowContext) -> _AutomationBenchRow:
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
        if not isinstance(input_payload, Mapping):
            raise TypeError("input must be a JSON object")
        if not isinstance(expected, Mapping):
            raise TypeError("expected must be a JSON object")
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a JSON object")

        task_name = input_payload.get("task_name")
        if not isinstance(task_name, str) or not task_name.strip():
            raise ValueError("input.task_name must be a non-empty string")
        for field_name in METRIC_FIELDS:
            value = expected.get(field_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"expected.{field_name} must be numeric")
            if float(value) != 1.0:
                raise ValueError(f"expected.{field_name} must be 1.0")

        return _AutomationBenchRow(
            id=row_id,
            input={"task_name": task_name},
            expected={name: float(expected[name]) for name in METRIC_FIELDS},
            metadata=dict(metadata),
            group_key=str(raw.get("group_key") or metadata.get("domain") or "default"),
        )

    def iter_cases(self, row: _AutomationBenchRow, context: DatasetRowContext) -> Iterable[Case]:
        del context
        yield Case(
            input=row.input,
            case_id=row.id,
            ground_truth=row.expected,
            group_key=row.group_key,
            metadata=row.metadata,
        )


def _find_training_sample(task_name: str) -> Any:
    from automationbench_skills.data import load_split

    try:
        return next(sample for sample in load_split("train") if sample.task_name == task_name)
    except StopIteration as error:
        raise ValueError(f"Task {task_name!r} is not in AutomationBench's frozen train split") from error


@contextmanager
def _temporary_environment(name: str, value: str) -> Iterable[None]:
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


async def _run_case(*, case: Case, targets: None, runtime: Any) -> CaseResult:
    """Run one real AutomationBench rollout against the candidate skills."""

    from automationbench_skills.runner import ModelSpec, run_one_async

    del targets
    task_name = str(case.input["task_name"])
    sample = _find_training_sample(task_name)
    skills_dir = Path("skills")

    model = ModelSpec()
    gateway_key: str | None = None
    provider = "openai"
    if runtime.model:
        target = inference_target(runtime)
        provider = target.model.partition(":")[0] or "beaker"
        gateway_key = target.api_key
        model = ModelSpec(
            name=target.model,
            base_url=target.base_url,
            api_key_var=GATEWAY_API_KEY_VAR,
            api="chat_completions",
        )

    try:
        with runtime.trace.model_call(
            operation="automationbench.agent_rollout",
            provider=provider,
            model=model.name,
            input_messages=sample.prompt,
        ) as model_call:
            if gateway_key is None:
                result = await run_one_async(sample, model=model, skills_dir=skills_dir)
            else:
                with _temporary_environment(GATEWAY_API_KEY_VAR, gateway_key):
                    result = await run_one_async(sample, model=model, skills_dir=skills_dir)
            model_call.output(result.trajectory)
    except Exception as error:
        return CaseResult.failed(str(error), retryable=True)

    if result.error is not None:
        return CaseResult.failed(
            str(result.error),
            context={
                "task_name": result.task_name,
                "domain": result.domain,
                "trajectory": result.trajectory,
            },
        )

    return CaseResult(
        output={
            "partial_credit": result.partial_credit,
            "task_completed_correctly": result.task_completed_correctly,
        },
        context={
            "task_name": result.task_name,
            "domain": result.domain,
            "trajectory": result.trajectory,
            "assertion_results": result.assertion_results,
            "end_state": result.end_state,
        },
    )


class _AutomationBenchScorer:
    """Use the benchmark's deterministic rollout metrics as Beaker scores."""

    async def score_case(self, *, case: Case, result: CaseResult) -> CaseScore:
        if OBJECTIVE_FIELD not in METRIC_FIELDS:
            raise RuntimeError("Select partial_credit or task_completed_correctly as OBJECTIVE_FIELD")
        output = result.output if isinstance(result.output, Mapping) else {}
        scores = {name: max(0.0, min(1.0, float(output.get(name, 0.0)))) for name in METRIC_FIELDS}
        return CaseScore(field_scores=scores, objective=scores[OBJECTIVE_FIELD], key=OBJECTIVE_FIELD)


@spec(
    dataset_schema=STANDARD_JSONL_CASE_SCHEMA,
    repository=("skills",),
)
def build_spec(ctx: OptimizationContext) -> Spec:
    """Optimize filesystem skills against deterministic AutomationBench tasks."""

    del ctx
    return Spec(
        data_loader=_AutomationBenchDataLoader(),
        run_case=_run_case,
        scorer=_AutomationBenchScorer(),
    )
