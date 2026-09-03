"""Beaker repository optimization spec for automationbench-skills.

Optimizes filesystem skills under ``skills/``. Each case is one AutomationBench
task identified by ``task_name``; the benchmark rubric scores the final world
state. Both ``task_completed_correctly`` and ``partial_credit`` are recorded;
the hill-climb objective is ``task_completed_correctly`` (strict pass) until a
different metric is confirmed.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from automationbench.clients import RetryingOpenAIChatCompletionsClient
from beaker import (
    Case,
    CaseDataLoader,
    CaseResult,
    CaseScore,
    DatasetRowContext,
    OptimizationContext,
    STANDARD_JSONL_CASE_SCHEMA,
    Spec,
    inference_target,
    spec,
)
from beaker.tracing import current_trace
from verifiers.types import ClientConfig

from automationbench_skills.data import load_samples
from automationbench_skills.runner import ModelSpec, get_client, run_one_async
from automationbench_skills.vendored.model_setup import resolve_api_key_var


OBJECTIVE_FIELD = "task_completed_correctly"
FIELD_NAMES = ("task_completed_correctly", "partial_credit")


class TracedChatCompletionsClient(RetryingOpenAIChatCompletionsClient):
    """Type-preserving adapter: same public client, traced native request."""

    async def get_native_response(
        self,
        prompt: Any,
        model: str,
        sampling_args: Any,
        tools: Any = None,
        **kwargs: Any,
    ) -> Any:
        trace = current_trace()
        with trace.model_call(
            operation="chat.completions",
            provider="openai",
            model=str(model),
            input_messages=prompt,
        ) as call:
            response = await super().get_native_response(prompt, model, sampling_args, tools, **kwargs)
            call.output(response)
            usage = getattr(response, "usage", None)
            if usage is not None:
                call.usage(
                    input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                )
            return response


@dataclass(frozen=True)
class _TaskRow:
    id: str
    input: dict[str, Any]
    expected: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    group_key: str = "default"


class _TaskDataLoader(CaseDataLoader[_TaskRow]):
    """JSONL rows keyed by frozen AutomationBench ``task_name``."""

    dataset_schema = STANDARD_JSONL_CASE_SCHEMA

    def parse_row(self, raw: Mapping[str, Any], context: DatasetRowContext) -> _TaskRow:
        del context
        missing = [name for name in ("id", "input", "expected") if name not in raw]
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
        task_name = str(input_payload.get("task_name") or row_id).strip()
        if not task_name:
            raise ValueError("input.task_name must be non-empty")
        return _TaskRow(
            id=row_id,
            input={"task_name": task_name, **dict(input_payload)},
            expected=dict(expected),
            metadata=dict(metadata),
            group_key=str(raw.get("group_key") or metadata.get("domain") or "default"),
        )

    def iter_cases(self, row: _TaskRow, context: DatasetRowContext) -> Iterable[Case]:
        del context
        yield Case(
            input=row.input,
            case_id=row.id,
            ground_truth=row.expected,
            group_key=row.group_key,
            metadata=row.metadata,
        )


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _skills_dir() -> Path:
    for base in (Path.cwd(), *Path.cwd().parents):
        candidate = base / "skills"
        if candidate.is_dir() and (base / "src" / "automationbench_skills").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the candidate skills/ directory")


def _eval_model(runtime: Any) -> ModelSpec:
    if getattr(runtime, "model", None):
        target = inference_target(runtime)
        os.environ["BEAKER_INFERENCE_API_KEY"] = target.api_key
        return ModelSpec(
            name=target.model,
            base_url=target.base_url,
            api_key_var="BEAKER_INFERENCE_API_KEY",
            api="chat_completions",
        )
    return ModelSpec()


def _eval_client(model: ModelSpec) -> Any:
    resolved = model.resolved_api()
    if resolved != "chat_completions":
        return get_client(model)
    key_var = resolve_api_key_var(resolved, model.api_key_var)
    if not os.environ.get(key_var):
        raise ValueError(f"No API key found. Set the {key_var} environment variable.")
    config = ClientConfig(
        api_key_var=key_var,
        api_base_url=model.base_url or "https://api.openai.com/v1",
        extra_headers={},
    )
    return TracedChatCompletionsClient(config)


async def _run_case(*, case: Case, targets: None, runtime: Any) -> CaseResult:
    del targets
    task_name = str((case.input or {}).get("task_name") or case.case_id or "").strip()
    if not task_name:
        return CaseResult.failed("Case input is missing task_name", retryable=False)
    by_name = {sample.task_name: sample for sample in load_samples()}
    sample = by_name.get(task_name)
    if sample is None:
        return CaseResult.failed(f"Unknown AutomationBench task {task_name!r}", retryable=False)

    model = _eval_model(runtime)
    skills_dir = _skills_dir()
    with runtime.trace.stage(
        "automationbench.run_one",
        inputs={"case_id": case.case_id, "task_name": task_name, "model": model.name},
    ) as stage:
        try:
            result = await run_one_async(
                sample,
                model=model,
                skills_dir=skills_dir,
                client=_eval_client(model),
            )
        except Exception as exc:
            return CaseResult.failed(str(exc), retryable=True)
        stage.output(
            {
                "task_completed_correctly": result.task_completed_correctly,
                "partial_credit": result.partial_credit,
            }
        )

    if result.error and not result.trajectory:
        return CaseResult.failed(str(result.error), retryable=str(result.error).startswith("timeout"))

    return CaseResult(
        output={
            "task_completed_correctly": float(result.task_completed_correctly),
            "partial_credit": float(result.partial_credit),
        },
        context={
            "task_name": result.task_name,
            "domain": result.domain,
            "trajectory": _jsonable(result.trajectory),
            "end_state": _jsonable(result.end_state),
            "assertion_results": _jsonable(result.assertion_results),
            "error": str(result.error) if result.error is not None else None,
        },
    )


class _RubricScorer:
    """Pass through the benchmark's deterministic rubric metrics."""

    async def score_case(self, *, case: Case, result: CaseResult) -> CaseScore:
        del case
        output = result.output if isinstance(result.output, Mapping) else {}
        scores = {name: float(output.get(name) or 0.0) for name in FIELD_NAMES}
        return CaseScore(
            field_scores=scores,
            objective=float(scores.get(OBJECTIVE_FIELD) or 0.0),
            key="default",
        )


@spec(
    dataset_schema=STANDARD_JSONL_CASE_SCHEMA,
    repository=("skills",),
)
def build_spec(ctx: OptimizationContext) -> Spec:
    del ctx
    return Spec(
        data_loader=_TaskDataLoader(),
        run_case=_run_case,
        scorer=_RubricScorer(),
    )
