"""Beaker repository optimization spec for AutomationBench skills.

The candidate is the live ``skills/`` tree. Evaluation policy stays here:
load a frozen-split task by name, run the harness ``run_one`` path, and score
the benchmark's deterministic assertion metrics.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from automationbench.clients import RetryingOpenAIChatCompletionsClient
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
    objective_score,
    spec,
)
from beaker.sdk.utils import to_json_safe
from beaker.tracing import current_trace
from verifiers.types import ClientConfig, RolloutInput

from automationbench_skills.data.tasks import Sample, load_samples
from automationbench_skills.runner import DEFAULT_MAX_STEPS, STATE_COLUMNS, ModelSpec, get_env
from automationbench_skills.skills_tools import set_skills_dir
from automationbench_skills.vendored.model_setup import build_sampling_args


FIELD_NAMES = ["task_completed_correctly", "partial_credit"]
# Hill-climb on assertion partial credit. Strict pass rate is still recorded
# as a field score but does not drive the optimizer objective.
FIELD_WEIGHTS = {"partial_credit": 1.0, "task_completed_correctly": 0.0}
DEFAULT_TIMEOUT_SECONDS = 180.0
_GATEWAY_API_KEY_VAR = "OPENAI_API_KEY"


@dataclass(frozen=True)
class _TaskRow:
    id: str
    input: dict[str, Any]
    expected: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    group_key: str = "default"


class _TaskDataLoader(CaseDataLoader[_TaskRow]):
    """JSONL rows keyed by AutomationBench ``task_name``."""

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


class _TracedChatCompletionsClient(RetryingOpenAIChatCompletionsClient):
    """Same client the harness already accepts, with Beaker spans on each call."""

    async def get_native_response(
        self, prompt: Any, model: str, sampling_args: Any, tools: Any = None, **kwargs: Any
    ) -> Any:
        with current_trace().model_call(
            operation="chat.completions",
            provider="openai",
            model=str(model),
            input_messages=prompt,
        ) as call:
            response = await super().get_native_response(prompt, model, sampling_args, tools, **kwargs)
            call.output(to_json_safe(response))
            return response


@cache
def _samples_by_name() -> dict[str, Sample]:
    return {sample.task_name: sample for sample in load_samples()}


def _skills_dir() -> Path:
    for candidate in (Path.cwd() / "skills", Path(__file__).resolve().parents[1] / "skills"):
        if candidate.is_dir():
            return candidate
    return Path.cwd() / "skills"


def _sample_for_case(case: Case) -> Sample:
    payload = case.input if isinstance(case.input, Mapping) else {}
    task_name = str(payload.get("task_name") or case.case_id).strip()
    sample = _samples_by_name().get(task_name)
    if sample is None:
        raise KeyError(f"unknown AutomationBench task_name {task_name!r}")
    return sample


def _model_for_runtime(runtime: Any) -> ModelSpec:
    """Gateway-routed ModelSpec when the run selected a model; else app defaults."""
    if getattr(runtime, "model", None):
        target = inference_target(runtime)
        os.environ[_GATEWAY_API_KEY_VAR] = target.api_key
        return ModelSpec(
            name=target.model,
            base_url=target.base_url,
            api_key_var=_GATEWAY_API_KEY_VAR,
            api="chat_completions",
        )
    return ModelSpec()


def _traced_client(model: ModelSpec) -> _TracedChatCompletionsClient:
    resolved = model.resolved_api() if model.api != "chat_completions" else "chat_completions"
    if resolved != "chat_completions":
        raise RuntimeError(f"Beaker evaluation expects the OpenAI Chat Completions client; got {resolved!r}.")
    if not os.environ.get(model.api_key_var):
        raise ValueError(f"No API key found. Set the {model.api_key_var} environment variable.")
    return _TracedChatCompletionsClient(
        ClientConfig(
            api_key_var=model.api_key_var,
            api_base_url=model.base_url or "https://api.openai.com/v1",
        )
    )


async def _run_case(*, case: Case, targets: None, runtime: Any) -> CaseResult:
    del targets
    try:
        sample = _sample_for_case(case)
    except KeyError as exc:
        return CaseResult.failed(str(exc), retryable=False)

    model = _model_for_runtime(runtime)
    skills_dir = _skills_dir()
    with runtime.trace.stage(
        "automationbench.run_one",
        inputs={"case_id": case.case_id, "task_name": sample.task_name, "skills_dir": str(skills_dir)},
    ) as stage:
        try:
            env = get_env(toolset="zapier", skills=True, max_steps=DEFAULT_MAX_STEPS)
            set_skills_dir(skills_dir)
            client = _traced_client(model)
            sampling_args = build_sampling_args(
                model.name, "chat_completions", model.reasoning_effort, model.extra_body
            )
            rollout = env.run_rollout(
                RolloutInput(
                    prompt=sample.prompt,
                    example_id=sample.index,
                    answer=sample.answer,
                    info=sample.info,
                ),
                client,
                model.name,
                sampling_args or {},
                state_columns=STATE_COLUMNS,
            )
            output = await asyncio.wait_for(rollout, DEFAULT_TIMEOUT_SECONDS)
            metrics = output.get("metrics") or {}
            partial = float(metrics.get("partial_credit", output.get("reward", 0.0)))
            strict = float(metrics.get("task_completed_correctly", 1.0 if partial == 1.0 else 0.0))
            completion = output.get("completion") or []
            trajectory = [m if isinstance(m, dict) else m.model_dump(mode="json") for m in completion]
            result_error = output.get("error")
            assertion_results = output.get("_assertion_results") or []
            end_state = output.get("_end_state")
        except TimeoutError:
            prediction = dict.fromkeys(FIELD_NAMES, 0.0)
            stage.output(prediction)
            return CaseResult(
                output=prediction,
                context={"error": f"timeout after {DEFAULT_TIMEOUT_SECONDS}s", "task_name": sample.task_name},
            )
        except Exception as exc:
            return CaseResult.failed(str(exc), retryable=True)

        prediction = {
            "task_completed_correctly": strict,
            "partial_credit": partial,
        }
        stage.output(prediction)
        return CaseResult(
            output=prediction,
            context={
                "task_name": sample.task_name,
                "domain": sample.domain,
                "error": None if result_error is None else str(result_error),
                "assertion_results": to_json_safe(assertion_results),
                "trajectory": to_json_safe(trajectory),
                "end_state": to_json_safe(end_state),
            },
        )


class _AssertionScorer:
    """Pass through the harness's deterministic rubric scores."""

    async def score_case(self, *, case: Case, result: CaseResult) -> CaseScore:
        del case
        output = result.output if isinstance(result.output, Mapping) else {}
        scores = {name: float(output.get(name) or 0.0) for name in FIELD_NAMES}
        return CaseScore(
            field_scores=scores,
            objective=objective_score(scores, field_weights=FIELD_WEIGHTS),
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
        scorer=_AssertionScorer(),
    )
