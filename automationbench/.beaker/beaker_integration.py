"""Beaker repository Integration for AutomationBench skills.

The candidate is the live ``skills/`` tree and the agent's system prompt,
``prompts/system.md``. Evaluation policy stays here: load a frozen-split task
by name, run the harness ``run_one`` path, and score the benchmark's
deterministic assertion metrics.

Contract: the dataset row's ``expected`` holds the task's assertion list
(``{"assertions": [...]}``), ``run_case`` returns no answer and reports the
final world state in ``CaseResult.context["end_state"]``, and ``score_case``
runs the benchmark's own ``partial_credit`` rubric against that state,
emitting one ``Check`` per assertion so the optimizer sees which requirements
failed.

Assertions reference simulated records by opaque id (``"contact_id":
"003xx000004MNO1"``); the scorer resolves those against the initial and end
world state so each check reads as ``salesforce_campaign_member_exists · David
Park · Q1 Product Launch Webinar`` rather than a dict of ids.

Model calls are traced by subclassing the verifiers client (see
``_TracedChatCompletionsClient``); the verifiers rollout is what talks to the
model, so there is no framework integration to enable here.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Mapping
from functools import cache
from pathlib import Path
from typing import Any

import verifiers as vf
from automationbench.clients import RetryingOpenAIChatCompletionsClient
from automationbench.rubric import partial_credit
from automationbench.rubric.registry import AssertionRegistry
from automationbench.schema.world import WorldState
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
    objective_score,
    repository,
)
from beaker.sdk.utils import to_json_safe
from beaker.tracing import current_trace
from pydantic import BaseModel, Field, field_validator
from verifiers.legacy.utils.error_utils import error_from_data, is_error_data
from verifiers.types import ClientConfig, RolloutInput

from automationbench_skills.data.tasks import Sample, load_samples
from automationbench_skills.prompts import load_system_prompt, with_system_prompt
from automationbench_skills.runner import (
    DEFAULT_MAX_STEPS,
    STATE_COLUMNS,
    TIMEOUT_GRACE_SECONDS,
    ModelSpec,
    get_env,
)
from automationbench_skills.skills_tools import set_skills_dir
from automationbench_skills.vendored.model_setup import build_sampling_args


# Hill-climb on assertion partial credit. Strict pass rate is still recorded
# as a field score but does not drive the optimizer objective.
FIELD_WEIGHTS = {"partial_credit": 1.0, "task_completed_correctly": 0.0}
DEFAULT_TIMEOUT_SECONDS = 600.0
_GATEWAY_API_KEY_VAR = "OPENAI_API_KEY"


class TaskRow(BaseModel):
    """JSONL row keyed by AutomationBench ``task_name`` (see ``upload_splits.py``)."""

    id: str = Field(min_length=1)
    input: dict[str, JsonValue]
    expected: dict[str, JsonValue]
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("expected")
    @classmethod
    def _assertions_present(cls, expected: dict[str, JsonValue]) -> dict[str, JsonValue]:
        assertions = expected.get("assertions")
        if not isinstance(assertions, list) or not all(
            isinstance(a, Mapping) and isinstance(a.get("type"), str) for a in assertions
        ):
            raise ValueError(
                "expected.assertions must be a list of assertion specs with a 'type' "
                "(re-run .beaker/upload_splits.py to build the dataset)"
            )
        return expected

    def task_name(self) -> str:
        task_name = str(self.input.get("task_name") or self.id).strip()
        if not task_name:
            raise ValueError("input.task_name must be non-empty")
        return task_name


class TaskSetup(RepositoryRunSetup[TaskRow]):
    row_model = TaskRow

    async def load_cases(self, row: TaskRow, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        del runtime
        yield Case(
            id=row.id,
            input={**row.input, "task_name": row.task_name()},
            expected=row.expected,
            metadata=row.metadata,
        )


class _TracedChatCompletionsClient(RetryingOpenAIChatCompletionsClient):
    """Same client the harness already accepts, with Beaker spans on each call.

    verifiers drives the agent loop itself and takes the client object as a
    parameter, so Beaker's LiteLLM/OpenAI/Anthropic integrations never see the
    calls. Subclassing keeps the exact type the harness checks for and wraps
    the one method every turn goes through: each span carries the full message
    list for that turn (tool calls and simulated tool results included, since
    they come back as messages) and the provider response, which is where the
    run's token usage and model-call counts come from. Only the candidate's
    calls are traced; the assertion rubric is deterministic and makes none.
    """

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


def _candidate_root() -> Path:
    """Directory holding the ``skills/`` and ``prompts/`` trees."""
    for candidate in (Path.cwd(), Path(__file__).resolve().parents[1]):
        if (candidate / "skills").is_dir():
            return candidate
    return Path.cwd()


def _sample_for(case_input: JsonValue, *, fallback_id: str | None = None) -> Sample:
    payload = case_input if isinstance(case_input, Mapping) else {}
    task_name = str(payload.get("task_name") or fallback_id or "").strip()
    sample = _samples_by_name().get(task_name)
    if sample is None:
        raise KeyError(f"unknown AutomationBench task_name {task_name!r}")
    return sample


def _model_for_runtime(runtime: RolloutRuntime[Any]) -> ModelSpec:
    """Gateway-routed ModelSpec when the run selected a model; else app defaults.

    The gateway applies the run's selected reasoning effort to requests that
    omit it, so no ``reasoning_effort`` is sent on that path.
    """
    if runtime.model:
        target = inference_target(runtime)
        os.environ[_GATEWAY_API_KEY_VAR] = target.api_key
        return ModelSpec(
            name=target.model,
            base_url=target.base_url,
            api_key_var=_GATEWAY_API_KEY_VAR,
            api="chat_completions",
            reasoning_effort=None,
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


def _rollout_error(raw: Any) -> BaseException | None:
    # ``run_rollout`` returns the serialized ``ErrorData`` mapping, not the
    # exception; rebuild the most specific ``vf.Error`` from its error chain.
    if raw is None:
        return None
    if isinstance(raw, BaseException):
        return raw
    if is_error_data(raw):
        return error_from_data(raw)
    return vf.Error(str(raw))


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime[Any]) -> CaseResult:
    sample = _sample_for(case_input)
    model = _model_for_runtime(runtime)
    root = _candidate_root()
    skills_dir = root / "skills"
    prompts_dir = root / "prompts"
    system_prompt = load_system_prompt(prompts_dir)
    with runtime.trace.stage(
        "automationbench.run_one",
        inputs={
            "task_name": sample.task_name,
            "skills_dir": str(skills_dir),
            "prompts_dir": str(prompts_dir),
            "system_prompt_chars": len(system_prompt or ""),
        },
    ) as stage:
        try:
            env = get_env(
                toolset="zapier",
                skills=True,
                max_steps=DEFAULT_MAX_STEPS,
                timeout=DEFAULT_TIMEOUT_SECONDS,
            )
            set_skills_dir(skills_dir)
            client = _traced_client(model)
            sampling_args = build_sampling_args(
                model.name, "chat_completions", model.reasoning_effort, model.extra_body
            )
            rollout = env.run_rollout(
                RolloutInput(
                    prompt=with_system_prompt(sample.prompt, system_prompt),
                    example_id=sample.index,
                    answer=sample.answer,
                    info=sample.info,
                ),
                client,
                model.name,
                sampling_args or {},
                state_columns=STATE_COLUMNS,
            )
            # The env stops its own loop at ``DEFAULT_TIMEOUT_SECONDS`` and
            # scores the world as the agent left it; this guard only catches a
            # rollout stuck outside that loop.
            output = await asyncio.wait_for(rollout, DEFAULT_TIMEOUT_SECONDS + TIMEOUT_GRACE_SECONDS)
        except TimeoutError:
            timeout = f"timeout after {DEFAULT_TIMEOUT_SECONDS + TIMEOUT_GRACE_SECONDS}s"
            stage.output({"error": timeout})
            return CaseResult(
                output=None,
                output_kind="none",
                context={"task_name": sample.task_name, "domain": sample.domain, "error": timeout},
            )
        except Exception as exc:
            raise RetryableCaseError(f"{type(exc).__name__}: {exc}") from exc

        completion = output.get("completion") or []
        result_error = _rollout_error(output.get("error"))
        end_state = output.get("_end_state")
        # verifiers swallows rollout exceptions into ``state["error"]`` and
        # still grades the untouched world. A model/provider/infra failure
        # means the agent never got to act, so the case did not run; an
        # agent-side failure (bad tool call, overlong prompt) is the
        # candidate's fault and keeps its earned score.
        if isinstance(result_error, vf.ModelError | vf.InfraError):
            message = f"{type(result_error).__name__}: {result_error}"
            stage.output({"error": message})
            raise RetryableCaseError(message) from result_error

        error = None if result_error is None else f"{type(result_error).__name__}: {result_error}"
        # The scorer needs the error and the end state; the model/tool turns are
        # already in the trace via ``_TracedChatCompletionsClient``.
        stage.output({"error": error, "messages": len(completion)})
        return CaseResult(
            output=None,
            output_kind="none",
            context={
                "task_name": sample.task_name,
                "domain": sample.domain,
                "error": error,
                "end_state": to_json_safe(end_state),
            },
        )


_SERVICE_FIELDS = sorted((str(f) for f in WorldState.model_fields if f != "meta"), key=len, reverse=True)


def _service_for(assertion_type: str) -> str | None:
    """WorldState service an assertion type targets (``gmail_message_sent_to`` -> ``gmail``).

    Falls back to the type's first token when it names an app split over several
    services (``facebook_page_post_exists`` -> ``facebook``).
    """
    for service in _SERVICE_FIELDS:
        if assertion_type == service or assertion_type.startswith(service + "_"):
            return service
    head = assertion_type.split("_", 1)[0]
    return head if any(service.startswith(head + "_") for service in _SERVICE_FIELDS) else None


def _assertions_for(case: Case) -> list[dict[str, Any]]:
    expected = case.expected if isinstance(case.expected, Mapping) else {}
    assertions = expected.get("assertions")
    return [dict(a) for a in assertions if isinstance(a, Mapping)] if isinstance(assertions, list) else []


# Simulated records carry ``id``; the readable field varies by app.
_LABEL_FIELDS = ("name", "title", "subject", "campaign_name", "account_name", "summary", "topic", "text", "email")
_LABEL_MAX_CHARS = 60


def _entity_index(*states: Any) -> dict[str, str]:
    """``record id -> readable label`` for every record in the given world states.

    Earlier states win, so a record renamed by the agent keeps the task author's name.
    """
    index: dict[str, str] = {}

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            label = " ".join(str(node[k]) for k in ("first_name", "last_name") if node.get(k)) or next(
                (node[k].strip() for k in _LABEL_FIELDS if isinstance(node.get(k), str) and node[k].strip()), ""
            )
            if label and isinstance(node.get("id"), str | int):
                index.setdefault(str(node["id"]), label)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for state in states:
        walk(state)
    return index


def _clip(text: str) -> str:
    return text if len(text) <= _LABEL_MAX_CHARS else text[: _LABEL_MAX_CHARS - 1] + "…"


def _check(
    outcome: Mapping[str, Any], *, error: str | None, entities: Mapping[str, str], held_initially: bool = False
) -> Check:
    assertion_type = str(outcome["type"])
    params = dict(outcome.get("params") or {})
    passed = bool(outcome.get("passed"))
    excluded = bool(outcome.get("excluded"))

    # ``type · param values``; an id is swapped for the record's label when the
    # world state knows it ("David Park" rather than "003xx000004MNO1").
    rendered = [
        entities.get(str(v), v if isinstance(v, str) else json.dumps(to_json_safe(v), default=str))
        for k, v in params.items()
        if k not in ("scored", "excluded")
    ]
    name = " · ".join([assertion_type, *(_clip(v) for v in rendered)])
    if excluded:
        message = (
            "excluded from scoring by the task author"
            if params.get("scored") is False or params.get("excluded") is True
            else "already satisfied in the initial state; excluded from scoring"
        )
    elif passed:
        message = None
    else:
        message = (
            "satisfied in the initial state; broken by the run" if held_initially else "not satisfied by the end state"
        ) + (f" (rollout error: {error})" if error else "")
    return Check(
        name=name,
        verdict="pass" if passed else "fail",
        message=message,
        group=_service_for(assertion_type),
        informational=excluded,
    )


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    """Run the benchmark's assertion rubric on the trusted side.

    ``partial_credit`` is the benchmark's own scoring function (including its
    free-assertion exclusion against the task's initial state); it is fed the
    dataset's assertions and the end state the candidate reported.
    """
    del case_files_dir
    assertions = _assertions_for(case)
    context = result.context
    error = None if context.get("error") is None else str(context["error"])
    end_state = context.get("end_state")
    initial_state = _sample_for(case.input, fallback_id=case.id).info.get("initial_state") or {}
    entities = _entity_index(initial_state, end_state if isinstance(end_state, Mapping) else {})
    initial_world = WorldState(**initial_state) if initial_state else None
    held_initially = [
        initial_world is not None and bool(AssertionRegistry.check(initial_world, a)) for a in assertions
    ]
    if isinstance(end_state, Mapping):
        state: dict[str, Any] = {
            "info": {"assertions": assertions},
            "world": WorldState(**end_state),
            "initial_state": initial_state,
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
        checks=tuple(
            _check(outcome, error=error, entities=entities, held_initially=held)
            for outcome, held in zip(outcomes, held_initially, strict=True)
        ),
    )


integration = Integration(
    # Optimizer-editable paths, relative to source_dir. "src" lets it change
    # the harness too (tool exposure, search_tools output, retries): the
    # candidate's src/ is loaded fresh per evaluation, scoring stays trusted.
    # Regenerate the playbook after changing this.
    targets=repository(paths=("skills", "prompts", "src")),
    run_setup=TaskSetup,
    run_case=run_case,
    score_case=score_case,
)
