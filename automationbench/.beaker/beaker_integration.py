"""Beaker repository integration for AutomationBench with filesystem skills.

One case is one public AutomationBench task. ``run_case`` runs the recipe's own
rollout path (``runner.run_rollout_raw``) with the candidate's ``prompts/`` and
``skills/`` folders, and the benchmark's deterministic rubric scores the final
world state. Candidates may edit only ``prompts/`` and ``skills/``, so the task
builders and rubric that produce ``assertion_results`` stay fixed.

Objective: ``task_completed_correctly`` (strict pass/fail, the benchmark's pass
rate). ``partial_credit`` is reported as context only.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from beaker import (
    Case,
    CaseResult,
    CaseScore,
    Check,
    Integration,
    JsonValue,
    RepositoryRunSetup,
    RetryableCaseError,
    RolloutRuntime,
    SetupRuntime,
    inference_target,
    repository,
)
from pydantic import BaseModel, Field


# The candidate checkout's project root (this file lives in <root>/.beaker/).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = PROJECT_ROOT / "skills"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
# Bounds the rollout loop; on expiry the world the agent changed so far is still scored.
TASK_TIMEOUT_SECONDS = 900.0
MAX_CHECKS = 100


class Row(BaseModel):
    id: str = Field(min_length=1)
    task_name: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    assertions: list[dict[str, JsonValue]] = Field(min_length=1)


class Setup(RepositoryRunSetup[Row]):
    row_model = Row

    async def load_cases(self, row: Row, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        del runtime
        yield Case(
            id=row.id,
            input={"task_name": row.task_name, "domain": row.domain, "instruction": row.instruction},
            expected={"assertions": row.assertions},
            metadata={"domain": row.domain},
        )


def _traced_client_classes() -> dict[str, type]:
    """Subclasses of the harness's own clients that trace each model turn.

    verifiers drives the model itself and Beaker's verifiers integration records
    tool spans only, so model calls are wrapped here around the one request
    method every turn goes through. Subclassing keeps the client types the
    harness accepts.
    """
    from automationbench.clients import OpenAIResponsesClient
    from beaker.tracing import current_trace

    from automationbench_skills.clients import CostTrackingChatCompletionsClient

    def traced(base: type, operation: str) -> type:
        class Traced(base):  # type: ignore[misc, valid-type]
            async def get_native_response(self, prompt: Any, model: str, *args: Any, **kwargs: Any) -> Any:
                with current_trace().model_call(
                    operation=operation, provider="openai", model=model, input_messages=prompt
                ) as call:
                    response = await super().get_native_response(prompt, model, *args, **kwargs)
                    call.output(response)
                    return response

        Traced.__name__ = f"Traced{base.__name__}"
        return Traced

    return {
        "chat_completions": traced(CostTrackingChatCompletionsClient, "chat"),
        "responses": traced(OpenAIResponsesClient, "responses"),
    }


_CLIENTS: dict[tuple[Any, asyncio.AbstractEventLoop], Any] = {}


def _model_and_client(runtime: RolloutRuntime) -> tuple[Any, Any]:
    """The recipe's default model, or the run's selected model through Beaker's gateway."""
    from verifiers.types import ClientConfig

    from automationbench_skills.clients import DEFAULT_MODEL_REQUEST_TIMEOUT
    from automationbench_skills.runner import ModelSpec, get_client

    if runtime.model:
        target = inference_target(runtime)
        spec = ModelSpec(name=target.model, base_url=target.base_url, api="chat_completions")
        api_key_var = "BEAKER_INFERENCE_API_KEY"
    else:
        spec = ModelSpec()
        api_key_var = spec.effective_api_key_var()
    api = spec.resolved_api()
    classes = _traced_client_classes()
    if api not in classes or spec.is_openrouter():
        return spec, get_client(spec)

    loop = asyncio.get_running_loop()
    key = (spec, loop)
    if key not in _CLIENTS:
        config = ClientConfig(
            api_key_var=api_key_var,
            api_base_url=spec.effective_base_url() or "https://api.openai.com/v1",
            extra_headers={},
            max_retries=0,
            timeout=DEFAULT_MODEL_REQUEST_TIMEOUT,
        )
        _CLIENTS[key] = classes[api](config)
    return spec, _CLIENTS[key]


def _rollout_failure(error: Any) -> Exception | None:
    """Classify the error verifiers stored on the rollout (serialized as ErrorData)."""
    if error is None:
        return None
    import verifiers.legacy as vf
    from verifiers.legacy.utils.error_utils import error_from_data, is_error_data

    exc = error_from_data(error) if is_error_data(error) else vf.Error(str(error))
    if isinstance(exc, (vf.ModelError, vf.InfraError)):
        return exc
    return None


def _final_message(trajectory: list[dict[str, Any]]) -> str | None:
    for message in reversed(trajectory):
        if message.get("role") == "assistant" and isinstance(message.get("content"), str) and message["content"]:
            return message["content"]
    return None


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime) -> CaseResult:
    from automationbench_skills.data.tasks import load_samples
    from automationbench_skills.runner import _to_result, run_rollout_raw

    assert isinstance(case_input, dict)
    task_name = str(case_input["task_name"])
    sample = next((s for s in load_samples() if s.task_name == task_name), None)
    if sample is None:
        raise ValueError(f"Unknown AutomationBench task {task_name!r}")

    spec, client = _model_and_client(runtime)
    with runtime.trace.stage("automationbench.rollout") as stage:
        try:
            raw = await run_rollout_raw(
                sample,
                model=spec,
                client=client,
                skills_dir=SKILLS_DIR,
                prompts_dir=PROMPTS_DIR,
                timeout=TASK_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            raise RetryableCaseError(f"rollout outlived its {TASK_TIMEOUT_SECONDS:.0f}s timeout") from exc
        result = _to_result(sample, raw)
        failure = _rollout_failure(result.error)
        if failure is not None:
            # The model or infrastructure failed, so the agent never got to finish.
            raise RetryableCaseError(f"{type(failure).__name__}: {failure}") from failure
        output = {
            "task_name": task_name,
            "model": spec.name,
            "task_completed_correctly": result.task_completed_correctly,
            "partial_credit": result.partial_credit,
            "assertion_results": result.assertion_results,
            "final_message": _final_message(result.trajectory),
            "turns": sum(1 for m in result.trajectory if m.get("role") == "assistant"),
            "error": str(result.error) if result.error is not None else None,
        }
        stage.output({k: output[k] for k in ("task_completed_correctly", "partial_credit", "turns", "error")})
    return CaseResult(output=output, output_kind="record")


def _assertion_check(assertion: dict[str, Any]) -> Check:
    atype = str(assertion.get("type", "assertion"))
    params = assertion.get("params") or {}
    name = " · ".join([atype, *(str(v) for v in params.values())])
    excluded = bool(assertion.get("excluded"))
    passed = bool(assertion.get("passed"))
    return Check(
        name=name[:300],
        verdict="pass" if passed else "fail",
        message="Excluded from scoring: already true before the agent acted, or marked unscored by the task."
        if excluded
        else None,
        group=atype.split("_", 1)[0],
        informational=excluded,
    )


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    del case_files_dir
    output = result.output if isinstance(result.output, dict) else {}
    assertions = output.get("assertion_results") or []
    expected = case.expected if isinstance(case.expected, dict) else {}
    expected_count = len(expected.get("assertions") or [])

    strict = float(output.get("task_completed_correctly") or 0.0)
    partial = float(output.get("partial_credit") or 0.0)
    checks = [_assertion_check(a) for a in assertions if isinstance(a, dict)]
    if expected_count and len(assertions) != expected_count:
        # The rubric reports one result per task assertion; a mismatch means the
        # rollout ran a different task definition than the dataset row.
        strict, partial = 0.0, 0.0
        checks.insert(
            0,
            Check(
                name="Assertions evaluated",
                verdict="fail",
                expected=expected_count,
                predicted=len(assertions),
                message="The rubric did not evaluate the task's assertions.",
            ),
        )
    if output.get("error"):
        checks.insert(0, Check(name="Rollout error", verdict="fail", message=str(output["error"])[:2000]))
    if len(checks) > MAX_CHECKS:
        failed = [c for c in checks if c.verdict == "fail"]
        checks = (failed + [c for c in checks if c.verdict != "fail"])[:MAX_CHECKS]

    return CaseScore(
        objective=strict,
        field_scores={"task_completed_correctly": strict, "partial_credit": partial},
        checks=tuple(checks),
    )


integration = Integration(
    targets=repository(("prompts", "skills")),
    run_setup=Setup,
    run_case=run_case,
    score_case=score_case,
)
