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
world state so each check reads as ``salesforce_campaign_member_exists ·
contact_id=David Park · campaign_id=Q1 Product Launch Webinar`` rather than a
dict of ids. Each check also carries the assertion handler's docstring as
``description``, the raw assertion params as ``expected`` and, when it failed,
a before/after diff of the targeted app's records as ``predicted`` (see
``world_diff.py``) so a reader sees what the agent actually did there without
replaying the trace.

Model calls are traced by subclassing the verifiers client (see
``_TracedChatCompletionsClient``); the verifiers rollout is what talks to the
model, so there is no framework integration to enable here.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Mapping
from functools import cache
from pathlib import Path
from typing import Any

import verifiers as vf
from automationbench.clients import RetryingOpenAIChatCompletionsClient
from automationbench.domains import get_combined_dataset
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
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator
from verifiers.clients import OpenAIChatCompletionsClient
from verifiers.legacy.utils.error_utils import error_from_data, is_error_data
from verifiers.types import ClientConfig
from world_diff import ServiceDiffs, clip, service_for

from automationbench_skills.data.tasks import Sample, load_samples
from automationbench_skills.prompts import load_system_prompt
from automationbench_skills.runner import (
    DEFAULT_MAX_STEPS,
    TIMEOUT_GRACE_SECONDS,
    ModelSpec,
    run_rollout_raw,
)


# Hill-climb mostly on assertion partial credit, with strict task completion
# as a smaller term so fully-correct runs are preferred.
FIELD_WEIGHTS = {"partial_credit": 0.8, "task_completed_correctly": 0.2}
DEFAULT_TIMEOUT_SECONDS = 600.0
# The scored benchmark domains (``upload_splits.py`` draws from the same set).
_PUBLIC_DOMAINS = ("sales", "marketing", "operations", "support", "finance", "hr")


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


class _SpanPerRequest(OpenAIChatCompletionsClient):
    """One Beaker ``model_call`` span per provider request.

    Sits below ``RetryingOpenAIChatCompletionsClient`` in the MRO so its retry
    loop calls into here on every attempt: a retried request (empty response,
    5xx, dropped connection) gets its own span with its own usage or error,
    instead of only the final attempt being recorded.
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


class _TracedChatCompletionsClient(RetryingOpenAIChatCompletionsClient, _SpanPerRequest):
    """Same client the harness already accepts, with Beaker spans on each request.

    verifiers drives the agent loop itself and takes the client object as a
    parameter, so Beaker's LiteLLM/OpenAI/Anthropic integrations never see the
    calls. Subclassing keeps the exact type the harness checks for; each span
    carries the full message list for that turn (tool calls and simulated tool
    results included, since they come back as messages) and the provider
    response, which is where the run's token usage and model-call counts come
    from. Only the candidate's calls are traced; the assertion rubric is
    deterministic and makes none.
    """


@cache
def _samples_by_name() -> dict[str, Sample]:
    return {sample.task_name: sample for sample in load_samples()}


@cache
def _authored_initial_states() -> dict[str, dict[str, Any]]:
    """``task_name -> initial_state`` straight from the pinned benchmark package.

    ``partial_credit`` excludes assertions already satisfied in the initial
    world, so the scorer reads that world from the trusted dependency rather
    than through the optimizer-editable ``src/`` loader ``run_case`` uses.
    """
    states: dict[str, dict[str, Any]] = {}
    for row in get_combined_dataset(list(_PUBLIC_DOMAINS)):
        info = row["info"]
        if isinstance(info, str):
            info = json.loads(info)
        states[str(info["task_name"])] = dict(info.get("initial_state") or {})
    return states


def _candidate_root() -> Path:
    """Directory holding the ``skills/`` and ``prompts/`` trees."""
    for candidate in (Path.cwd(), Path(__file__).resolve().parents[1]):
        if (candidate / "skills").is_dir():
            return candidate
    return Path.cwd()


def _sample_for(case_input: JsonValue) -> Sample:
    payload = case_input if isinstance(case_input, Mapping) else {}
    task_name = str(payload.get("task_name") or "").strip()
    sample = _samples_by_name().get(task_name)
    if sample is None:
        raise KeyError(f"unknown AutomationBench task_name {task_name!r}")
    return sample


def _client_for(runtime: RolloutRuntime[Any]) -> tuple[_TracedChatCompletionsClient, ModelSpec]:
    """The model the run selected, through Beaker's inference gateway; else the app's own defaults.

    A run started with a model choice (model-swap runs) resolves to a gateway
    endpoint, a run-scoped token and a canonical model name; the gateway applies
    the run's reasoning effort, so none is sent. With no selection the app's
    default model and ``OPENAI_API_KEY`` are used as-is, which inside a hosted
    rollout is Beaker's provider proxy. No key is ever asked of the user.
    """
    if runtime.model:
        target = inference_target(runtime)
        model = ModelSpec(name=target.model, base_url=target.base_url, api="chat_completions", reasoning_effort=None)
        api_key: str | None = target.api_key
    else:
        model = ModelSpec()
        if model.resolved_api() != "chat_completions":
            raise RuntimeError(f"Beaker evaluation expects a Chat Completions model; got {model.resolved_api()!r}.")
        api_key = os.environ.get(model.api_key_var)
    # Same SDK-level retry and timeout settings as the harness's own verifiers client.
    sdk = ClientConfig(api_key_var=model.api_key_var)
    client = AsyncOpenAI(api_key=api_key, base_url=model.base_url, max_retries=sdk.max_retries, timeout=sdk.timeout)
    return _TracedChatCompletionsClient(client), model


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
        client, model = _client_for(runtime)
        try:
            # Same execution path as the app's own ``run_one_async``; only the
            # client (traced, run-scoped) and the scoring below are Beaker's.
            # The env stops its own loop at ``DEFAULT_TIMEOUT_SECONDS`` and
            # scores the world as the agent left it; the helper's watchdog only
            # catches a rollout stuck outside that loop, which leaves nothing
            # to grade.
            output = await run_rollout_raw(
                sample,
                model=model,
                client=client,
                skills_dir=skills_dir,
                prompts_dir=prompts_dir,
                max_steps=DEFAULT_MAX_STEPS,
                timeout=DEFAULT_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            timeout = f"timeout after {DEFAULT_TIMEOUT_SECONDS + TIMEOUT_GRACE_SECONDS}s"
            stage.output({"error": timeout})
            raise RetryableCaseError(timeout) from exc
        finally:
            await client.close()

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
        # The client retries provider failures for longer than the env's time
        # budget, so a run whose every request failed ends by the env's own
        # timeout with no completed turn (``completion`` is empty) and an
        # untouched world: the agent never acted.
        if output.get("stop_condition") == "timeout_reached" and not completion:
            message = "timed out before the first completed model turn"
            stage.output({"error": message})
            raise RetryableCaseError(message)

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


def _assertions_for(case: Case) -> list[dict[str, Any]]:
    expected = case.expected if isinstance(case.expected, Mapping) else {}
    assertions = expected.get("assertions")
    return [dict(a) for a in assertions if isinstance(a, Mapping)] if isinstance(assertions, list) else []


# Simulated records carry ``id``; the readable field varies by app.
_LABEL_FIELDS = ("name", "title", "subject", "campaign_name", "account_name", "summary", "topic", "text", "email")
_META_KEYS = ("scored", "excluded")  # assertion params about scoring, not the check itself


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


@cache
def _description_for(assertion_type: str) -> str | None:
    """First line of the benchmark handler's docstring, e.g. "Check if an email was sent to ..."."""
    handler = AssertionRegistry._handlers.get(assertion_type)
    doc = getattr(handler, "__doc__", None) if handler is not None else None
    if not isinstance(doc, str):
        return None
    return next((line.strip() for line in doc.splitlines() if line.strip()), None)


def _check(
    outcome: Mapping[str, Any],
    *,
    error: str | None,
    entities: Mapping[str, str],
    held_initially: bool = False,
    diffs: ServiceDiffs | None = None,
) -> Check:
    assertion_type = str(outcome["type"])
    params = dict(outcome.get("params") or {})
    passed = bool(outcome.get("passed"))
    excluded = bool(outcome.get("excluded"))
    service = service_for(assertion_type)

    # ``type · key=value``; an id is swapped for the record's label when the
    # world state knows it ("David Park" rather than "003xx000004MNO1").
    rendered = [
        f"{k}={clip(entities.get(str(v), v if isinstance(v, str) else json.dumps(to_json_safe(v), default=str)))}"
        for k, v in params.items()
        if k not in _META_KEYS
    ]
    name = " · ".join([assertion_type, *rendered])
    expected = {k: to_json_safe(v) for k, v in params.items() if k not in _META_KEYS}
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
    # Only failed, scored checks carry the diff: that is where a reader needs
    # to know what the agent did in the app without replaying the trace.
    predicted = diffs.get(service) if diffs is not None and not passed and not excluded else None
    return Check(
        name=name,
        verdict="pass" if passed else "fail",
        message=message,
        group=service,
        informational=excluded,
        description=_description_for(assertion_type),
        expected=expected,
        predicted=predicted,
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
    task_name = str((case.input if isinstance(case.input, Mapping) else {}).get("task_name") or case.id).strip()
    if task_name not in _authored_initial_states():
        raise KeyError(f"unknown AutomationBench task_name {task_name!r}")
    initial_state = _authored_initial_states()[task_name]
    entities = _entity_index(initial_state, end_state if isinstance(end_state, Mapping) else {})
    initial_world = WorldState(**initial_state) if initial_state else None
    held_initially = [
        initial_world is not None and bool(AssertionRegistry.check(initial_world, a)) for a in assertions
    ]
    diffs: ServiceDiffs | None = None
    if isinstance(end_state, Mapping):
        state: dict[str, Any] = {
            "info": {"assertions": assertions},
            "world": WorldState(**end_state),
            "initial_state": initial_state,
        }
        partial = float(partial_credit(state))
        outcomes: list[Mapping[str, Any]] = list(state.get("_assertion_results") or [])
        diffs = ServiceDiffs(initial_state, initial_world, end_state)
    elif not context:
        # Beaker sheds ``context`` when a case result is over its wire limit;
        # an absent context is not a failed case.
        raise RuntimeError("CaseResult.context is empty (dropped for size?); the end state cannot be scored")
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
            _check(outcome, error=error, entities=entities, held_initially=held, diffs=diffs)
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
