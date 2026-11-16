"""Beaker integration for AutomationBench skills.

Load each dataset row's frozen task, run the candidate, and score with PI's
assertions from ``expected``. Evaluation policy stays in this integration.

``run_case`` returns bounded app changes for display and the full world in
``CaseResult.context["end_state"]``. Spreadsheet write history travels in
``context["grading_state"]`` and is restored before ``score_case`` applies PI's
``partial_credit`` rubric. Final cell values alone cannot recover write history.

Each assertion produces a Check with its parameters and available grading
observations. The app diff is displayed once per case.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from functools import cache
from pathlib import Path
from typing import Any

from assertion_diagnostics import capture_assertion_returns, failure_trace
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
    RetryableCaseError,
    RolloutRuntime,
    RunSetup,
    SetupRuntime,
    inference_target,
    objective_score,
    repository,
)
from beaker.sdk.utils import to_json_safe
from case_fixture import PI_COMMIT, CaseFixture
from pydantic import BaseModel, Field, model_validator
from world_diff import ServiceDiffs, service_for

from automationbench_skills.data.tasks import BENCHMARK_DOMAINS
from automationbench_skills.grading_state import assertion_outcomes, restore_grading_state
from automationbench_skills.prompts import load_system_prompt
from automationbench_skills.runner import (
    DEFAULT_MAX_STEPS,
    ModelSpec,
    run_rollout_raw,
)
from automationbench_skills.scenario_clock import scenario_time


# Hill-climb mostly on assertion partial credit, with strict task completion
# as a smaller term so fully-correct runs are preferred.
FIELD_WEIGHTS = {"partial_credit": 0.8, "task_completed_correctly": 0.2}
DEFAULT_TIMEOUT_SECONDS = 2400.0
# Mirrors beaker_runtime.optimization.evaluation.repository.protocol's
# MAX_EXECUTION_PAYLOAD_BYTES (not exposed by beaker-sdk 0.6.4).
_RESULT_MAX_BYTES = 8 * 1024 * 1024
# The child adds an index, error and compact trace receipt after run_case.
# Reserve space for those fields before allocating any display output.
_RESULT_METADATA_RESERVE_BYTES = 64 * 1024


class TaskRow(BaseModel):
    """JSONL row keyed by AutomationBench ``task_name`` (see ``upload_splits.py``)."""

    id: str = Field(min_length=1)
    input: dict[str, JsonValue]
    expected: dict[str, JsonValue]
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    runtime_fixture: CaseFixture | None = None

    @model_validator(mode="after")
    def validate_fixture(self) -> TaskRow:
        task_name = str(self.input.get("task_name") or self.id).strip()
        if self.runtime_fixture is None:
            if "task_sha256" in self.metadata or "scenario_sha256" in self.metadata:
                raise ValueError("Frozen dataset row is missing runtime_fixture")
            if self.metadata.get("benchmark_commit") != PI_COMMIT:
                raise ValueError("Dataset benchmark commit does not match this PI recipe; re-upload the dataset")
            # Resolve standard rows from the trusted dependency. Frozen pilot rows
            # retain their own task contents, including pre-correction examples.
            task = _published_tasks().get(task_name)
            if task is None:
                raise ValueError(f"Unknown AutomationBench task_name: {task_name!r}")
            fixture = CaseFixture.model_validate(
                {
                    "task": task,
                    "scenario": {
                        "source": "pi",
                        "fallback_time": scenario_time(task["prompt"], task["info"]["initial_state"]),
                    },
                }
            )
            fixture.validate_reference(task_name, self.expected, {**self.metadata, **fixture.hashes()})
            self.runtime_fixture = fixture
        else:
            self.runtime_fixture.validate_reference(task_name, self.expected, self.metadata)
        return self


class TaskSetup(RunSetup[TaskRow]):
    row_model = TaskRow

    async def load_cases(self, row: TaskRow, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        del runtime
        assert row.runtime_fixture is not None  # Validated or resolved by TaskRow.
        yield Case(
            id=row.id,
            input={**row.input, "runtime_fixture": row.runtime_fixture.model_dump(mode="json")},
            expected=row.expected,
            metadata={**row.metadata, "runtime_benchmark_commit": PI_COMMIT, "tool_interface": "pi_api"},
        )


def _candidate_root() -> Path:
    """Directory holding the ``skills/`` and ``prompts/`` trees."""
    for candidate in (Path.cwd(), Path(__file__).resolve().parents[1]):
        if (candidate / "skills").is_dir():
            return candidate
    return Path.cwd()


@cache
def _published_tasks() -> dict[str, dict[str, Any]]:
    """Read the pinned dependency, independently of optimizer-editable loaders."""
    tasks = {}
    for row in get_combined_dataset(BENCHMARK_DOMAINS):
        info = json.loads(row["info"]) if isinstance(row["info"], str) else row["info"]
        tasks[info["task_name"]] = {**row, "info": info}
    return tasks


def _fixture_for(case_input: JsonValue) -> CaseFixture:
    """Execution and scoring use only the fixture frozen during setup."""
    payload = case_input if isinstance(case_input, Mapping) else {}
    if payload.get("runtime_fixture") is None:
        raise ValueError("Case is missing runtime_fixture; load the dataset through TaskSetup first")
    return CaseFixture.model_validate(payload["runtime_fixture"])


def default_model_spec() -> ModelSpec:
    """The model a run without a model choice evaluates, through Beaker's provider proxy.

    Pinned here rather than taken from ``ModelSpec()`` so that changing the
    app's (and CLI's) default model never changes what Beaker optimizes. Chat
    Completions requests use the native runtime and Beaker tracing.
    """
    return ModelSpec(name="gpt-5.6-luna", api="chat_completions", reasoning_effort="medium")


def _model_for(runtime: RolloutRuntime[Any]) -> tuple[ModelSpec, str | None]:
    if runtime.model:
        target = inference_target(runtime)
        return ModelSpec(name=target.model, base_url=target.base_url, reasoning_effort=None), target.api_key
    return default_model_spec(), None


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime[Any]) -> CaseResult:
    fixture = _fixture_for(case_input)
    sample = fixture.sample()
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
        model, api_key = _model_for(runtime)
        try:
            # The native runner is shared with the CLI. The gateway credential
            # stays in the host client and is not serialized into the task.
            output = await run_rollout_raw(
                sample,
                model=model,
                api_key=api_key,
                skills_dir=skills_dir,
                prompts_dir=prompts_dir,
                toolset="api",
                max_steps=DEFAULT_MAX_STEPS,
                timeout=DEFAULT_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            timeout = "Native runtime timed out"
            stage.output({"error": timeout})
            raise RetryableCaseError(timeout) from exc

        completion = output.get("completion") or []
        result_error = output.get("error")
        end_state = output.get("_end_state")
        # Native provider/runtime failures ask Beaker to retry the case.
        if result_error is not None:
            message = f"{result_error.type}: {result_error.message}"
            stage.output({"error": message})
            raise RetryableCaseError(message)
        # An untouched world after a stalled first turn is not a completed evaluation.
        if output.get("stop_condition") == "timeout_reached" and not completion:
            message = "timed out before the first completed model turn"
            stage.output({"error": message})
            raise RetryableCaseError(message)

        error = None if result_error is None else f"{result_error.type}: {result_error.message}"
        # The scorer needs the error and the end state; the model/tool turns are
        # already in the trace via the native runtime adapter.
        stage.output({"error": error, "messages": len(completion)})
        observed = None
        if isinstance(end_state, Mapping):
            initial = fixture.initial_state()
            initial_world = WorldState(**initial) if initial else None
            observed = ServiceDiffs(initial, initial_world, end_state).to_output()
        context: dict[str, JsonValue] = {
            "task_name": sample.task_name,
            "domain": sample.domain,
            "runtime_benchmark_commit": PI_COMMIT,
            "tool_interface": "pi_api",
            "trace_capture": fixture.scenario.trace_capture,
            "error": error,
            "end_state": to_json_safe(end_state),
            "grading_state": to_json_safe(output.get("_grading_state")),
        }
        display = _budget_output(to_json_safe(observed), context)
        return CaseResult(output=display, output_kind="record" if display is not None else "none", context=context)


def _budget_output(output: JsonValue, context: Mapping[str, JsonValue]) -> JsonValue:
    """Shrink display evidence before it can displace the scoring state."""
    if output is None:
        return None
    # Match write_execution_payload's compact JSON and default ASCII escaping;
    # UTF-8 size alone undercounts non-ASCII world states on this transport.
    envelope = {"context": context, "output": None, "output_kind": "record"}
    available = (
        _RESULT_MAX_BYTES
        - _RESULT_METADATA_RESERVE_BYTES
        - len(json.dumps(envelope, separators=(",", ":")).encode("utf-8"))
        + len("null")
    )
    omitted: JsonValue = {"summary": "App changes omitted to preserve the scoring state.", "truncated": True}
    for display in (output, omitted):
        if len(json.dumps(display, separators=(",", ":")).encode("utf-8")) <= available:
            return display
    # If even the marker cannot fit, return the same no-output result used
    # before app diffs were added. Never trim the context to make room.
    return None


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
    world: WorldState | None = None,
    diagnostic: str | None = None,
) -> Check:
    assertion_type = str(outcome["type"])
    params = dict(outcome.get("params") or {})
    passed = bool(outcome.get("passed"))
    excluded = bool(outcome.get("excluded"))
    service = service_for(assertion_type)

    # ``type · key=value``; an id is swapped for the record's label when the
    # world state knows it ("David Park" rather than "003xx000004MNO1"). Keep
    # values complete so distinct requirements cannot collapse to one name.
    rendered = [
        f"{k}={entities.get(str(v), v if isinstance(v, str) else json.dumps(to_json_safe(v), ensure_ascii=False, default=str))}"
        for k, v in params.items()
        if k not in _META_KEYS
    ]
    name = " · ".join([assertion_type, *rendered])
    requirements = {k: to_json_safe(v) for k, v in params.items() if k not in _META_KEYS}
    description = _description_for(assertion_type)
    if requirements:
        description = "\n".join(
            part
            for part in (description, "Condition parameters: " + json.dumps(requirements, ensure_ascii=False))
            if part
        )
    if excluded:
        message = outcome.get("exclusion_reason") or (
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
    if not passed and diagnostic:
        message += "\n\n" + diagnostic
    elif not passed and not excluded and world is not None:
        message += "; see OUTPUT for observed app changes"
    return Check(
        name=name,
        verdict="pass" if passed else "fail",
        message=message,
        group=service,
        informational=excluded,
        description=description,
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
    initial_state = _fixture_for(case.input).initial_state()
    entities = _entity_index(initial_state, end_state if isinstance(end_state, Mapping) else {})
    initial_world = WorldState(**initial_state) if initial_state else None
    world: WorldState | None = None
    diagnostics: Mapping[int, Any] = {}
    if isinstance(end_state, Mapping):
        world = WorldState(**end_state)
        restore_grading_state(world, context.get("grading_state"))
        state: dict[str, Any] = {
            "info": {"assertions": assertions},
            "world": world,
            "initial_state": initial_state,
        }
        with capture_assertion_returns(world, assertions) as diagnostics:
            partial = float(partial_credit(state))
        outcomes = assertion_outcomes(world, assertions, initial_world)
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
                "held_initially": initial_world is not None and bool(AssertionRegistry.check(initial_world, a)),
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
            _check(
                outcome,
                error=error,
                entities=entities,
                held_initially=outcome["held_initially"],
                world=world,
                diagnostic=failure_trace(assertion, diagnostics.get(id(assertion), ()))
                if world is not None and not outcome.get("passed")
                else None,
            )
            for outcome, assertion in zip(outcomes, assertions, strict=True)
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
