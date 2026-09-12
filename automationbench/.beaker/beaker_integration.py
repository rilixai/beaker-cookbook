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
a before/after diff of the targeted app's records as ``predicted`` so a reader
sees what the agent actually did there without replaying the trace.

Model calls are traced by subclassing the verifiers client (see
``_TracedChatCompletionsClient``); the verifiers rollout is what talks to the
model, so there is no framework integration to enable here.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Iterator, Mapping
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


def _clip(text: str, limit: int = _LABEL_MAX_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


@cache
def _description_for(assertion_type: str) -> str | None:
    """First line of the benchmark handler's docstring, e.g. "Check if an email was sent to ..."."""
    handler = AssertionRegistry._handlers.get(assertion_type)
    doc = getattr(handler, "__doc__", None) if handler is not None else None
    if not isinstance(doc, str):
        return None
    return next((line.strip() for line in doc.splitlines() if line.strip()), None)


# ``predicted`` on a failed check is a before/after diff of the app it targets.
# Records are reduced to their own fields and the diff is re-rendered with
# fewer records / shorter strings until it serializes under the byte cap.
_DIFF_MAX_RECORDS = 5  # per added/changed/removed list
_DIFF_MAX_STR_CHARS = 400
_DIFF_MAX_FIELDS = 40  # per record
_DIFF_MAX_LIST_ITEMS = 20  # per scalar list inside a record
_DIFF_MAX_BYTES = 8_000  # serialized ``predicted`` per check
_DIFF_SHRINK_STEPS = ((_DIFF_MAX_RECORDS, _DIFF_MAX_STR_CHARS), (3, 400), (2, 200), (1, 120), (0, 0))
_META_KEYS = ("scored", "excluded")

_Record = dict[str, Any]  # flattened record: scalars and lists of scalars
_Collection = dict[str, _Record]  # record key -> flattened record


class _Authored:
    """What the task author wrote for one record collection, per ``_authored``."""

    def __init__(self) -> None:
        self.fields_by_id: dict[str, set[str]] = {}  # author-given id -> field names the author set on it
        self.key_fields: set[str] | None = None  # field names every authored record has; None when unknown

    def add(self, raw_record: Mapping[str, Any]) -> str | None:
        fields = set(_flatten_record(raw_record))
        raw_id = _record_key(raw_record)
        if raw_id is not None:
            self.fields_by_id[raw_id] = fields
        self.key_fields = fields if self.key_fields is None else self.key_fields & fields
        return raw_id


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _is_collection(value: Any) -> bool:
    """A non-empty list of mappings: the world's record collections (messages, contacts, rows)."""
    return isinstance(value, list) and bool(value) and all(isinstance(item, Mapping) for item in value)


def _flatten_record(record: Mapping[str, Any]) -> _Record:
    """The record's own fields: scalars and scalar lists, nested mappings flattened to dotted keys.

    Nested collections are left out; they are diffed under their own path.
    """
    flat: _Record = {}

    def walk(mapping: Mapping[str, Any], prefix: str) -> None:
        for key, raw in mapping.items():
            value = to_json_safe(raw)
            name = f"{prefix}{key}"
            if _is_scalar(value):
                flat[name] = value
            elif isinstance(value, list) and all(_is_scalar(item) for item in value):
                flat[name] = value
            elif isinstance(value, Mapping):
                walk(value, name + ".")

    walk(record, "")
    return flat


def _record_key(record: Mapping[str, Any]) -> str | None:
    """Records match across states by ``id``; ``None`` when they have no usable one."""
    record_id = record.get("id")
    return str(record_id) if isinstance(record_id, str | int) and not isinstance(record_id, bool) else None


def _authored(initial: Any, raw: Any, path: str, out: dict[str, _Authored]) -> None:
    """Per collection path, the ids and fields the task author wrote in ``raw``.

    ``initial`` is the same state after ``WorldState`` validation, which fills
    defaults and generates ids and timestamps for what the author left out.
    Only the author's values are stable across the initial and end worlds; the
    run builds its own ``WorldState`` and regenerates everything else.
    """
    if isinstance(initial, Mapping):
        raw_map = raw if isinstance(raw, Mapping) else {}
        for key, value in initial.items():
            _authored(value, raw_map.get(key), f"{path}.{key}" if path else str(key), out)
    elif _is_collection(initial):
        raw_records = raw if isinstance(raw, list) else []
        authored = out.setdefault(path, _Authored())
        for position, record in enumerate(initial):
            raw_record = raw_records[position] if position < len(raw_records) else None
            raw_record = raw_record if isinstance(raw_record, Mapping) else {}
            raw_id = authored.add(raw_record)
            _authored(record, raw_record, f"{path}[{raw_id if raw_id is not None else position}]", out)


def _view(record: _Record, fields: set[str] | None) -> _Record:
    return record if fields is None else {k: v for k, v in record.items() if k in fields}


def _collections(node: Any, path: str, authored: Mapping[str, _Authored]) -> Iterator[tuple[str, _Collection]]:
    """Every record collection under ``node`` by path (``gmail.messages``, ``zendesk.tickets[t1].comments``).

    A record keeps its ``id`` as key when the author gave it; otherwise (no id,
    or one generated by the schema) it is keyed by its authored content, so it
    still shows up as added/removed but never as changed.
    """
    if isinstance(node, Mapping):
        for key, value in node.items():
            yield from _collections(value, f"{path}.{key}" if path else str(key), authored)
    elif _is_collection(node):
        known = authored.get(path, _Authored())
        records: _Collection = {}
        seen: dict[str, int] = {}
        for position, record in enumerate(node):
            record_id = _record_key(record)
            flat = _flatten_record(record)
            label: str | int
            if record_id is not None and record_id in known.fields_by_id:
                key, label = f"id:{record_id}", record_id
            else:
                content = {k: v for k, v in _view(flat, known.key_fields).items() if k != "id"}
                key, label = "json:" + json.dumps(content, sort_keys=True), position
                seen[key] = seen.get(key, 0) + 1
                key = f"{key}#{seen[key]}"
            records.setdefault(key, flat)
            yield from _collections(record, f"{path}[{label}]", authored)
        yield path, records


def _shrink(record: _Record, max_str: int) -> dict[str, Any]:
    kept: dict[str, Any] = {}
    for name, value in list(record.items())[:_DIFF_MAX_FIELDS]:
        if isinstance(value, str):
            kept[name] = _clip(value, max_str)
        elif isinstance(value, list):
            kept[name] = [_clip(v, max_str) if isinstance(v, str) else v for v in value[:_DIFF_MAX_LIST_ITEMS]]
            if len(value) > _DIFF_MAX_LIST_ITEMS:
                kept[name].append(f"… +{len(value) - _DIFF_MAX_LIST_ITEMS} more")
        else:
            kept[name] = value
    if len(record) > _DIFF_MAX_FIELDS:
        kept["truncated_fields"] = len(record) - _DIFF_MAX_FIELDS
    return kept


def _service_diff(
    service: str, fields: list[str], raw_initial: Mapping[str, Any], initial: Mapping[str, Any], end: Mapping[str, Any]
) -> dict[str, Any]:
    """Added / changed / removed records of one app between the initial and end world state.

    ``fields`` are the ``WorldState`` fields the app spans (``["gmail"]``, or
    ``["facebook_conversions", "facebook_lead_ads", "facebook_pages"]`` for
    ``facebook``); record paths start with the concrete field. ``initial`` and
    ``end`` are ``WorldState`` dumps; ``raw_initial`` is the task author's
    initial state, which decides what counts as a real change (see
    ``_authored``): an id-matched record is changed when a field the author set
    on it differs.
    """
    authored: dict[str, _Authored] = {}
    before: dict[str, _Collection] = {}
    after: dict[str, _Collection] = {}
    for field in fields:
        _authored(initial.get(field), raw_initial.get(field), field, authored)
        before.update(_collections(initial.get(field), field, authored))
        after.update(_collections(end.get(field), field, authored))
    added: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for path in sorted(before.keys() | after.keys()):
        old, new = before.get(path, {}), after.get(path, {})
        fields_by_id = authored[path].fields_by_id if path in authored else {}
        added.extend({"path": path, "record": new[k]} for k in new if k not in old)
        for record_id, fields in fields_by_id.items():
            key = f"id:{record_id}"
            if key in old and key in new and _view(old[key], fields) != _view(new[key], fields):
                changed.append({"path": path, "before": old[key], "after": new[key]})
        removed.extend({"path": path, "record": old[k]} for k in old if k not in new)

    def render(entries: list[dict[str, Any]], max_records: int, max_str: int) -> list[dict[str, Any]]:
        kept = [
            {k: _shrink(v, max_str) if isinstance(v, dict) else v for k, v in entry.items()}
            for entry in entries[:max_records]
        ]
        if len(entries) > max_records:
            kept.append({"truncated": len(entries) - max_records})
        return kept

    lists = {"added": added, "changed": changed, "removed": removed}
    if not any(lists.values()):
        return {"service": service, "added": [], "changed": [], "removed": []}
    for max_records, max_str in _DIFF_SHRINK_STEPS:
        diff: dict[str, Any] = {"service": service}
        diff.update({k: render(v, max_records, max_str) for k, v in lists.items() if v})
        if len(json.dumps(diff, ensure_ascii=False).encode("utf-8")) <= _DIFF_MAX_BYTES:
            break
    return diff


class _ServiceDiffs:
    """Per-service diff of the case's world, computed once and shared by every check targeting that app."""

    def __init__(
        self, raw_initial: Mapping[str, Any], initial_world: WorldState | None, end_state: Mapping[str, Any]
    ) -> None:
        self._raw_initial = raw_initial
        self._initial: Mapping[str, Any] = initial_world.model_dump(mode="json") if initial_world is not None else {}
        self._end = end_state
        self._cache: dict[str, dict[str, Any] | None] = {}

    def get(self, service: str | None) -> dict[str, Any] | None:
        if service is None:
            return None
        if service not in self._cache:
            spanned = (
                [service]
                if service in _SERVICE_FIELDS
                else [f for f in _SERVICE_FIELDS if f.startswith(service + "_")]
            )
            fields = sorted(f for f in spanned if f in self._raw_initial or f in self._end)
            self._cache[service] = (
                _service_diff(service, fields, self._raw_initial, self._initial, self._end) if fields else None
            )
        return self._cache[service]


def _check(
    outcome: Mapping[str, Any],
    *,
    error: str | None,
    entities: Mapping[str, str],
    held_initially: bool = False,
    diffs: _ServiceDiffs | None = None,
) -> Check:
    assertion_type = str(outcome["type"])
    params = dict(outcome.get("params") or {})
    passed = bool(outcome.get("passed"))
    excluded = bool(outcome.get("excluded"))
    service = _service_for(assertion_type)

    # ``type · key=value``; an id is swapped for the record's label when the
    # world state knows it ("David Park" rather than "003xx000004MNO1").
    rendered = [
        f"{k}={_clip(entities.get(str(v), v if isinstance(v, str) else json.dumps(to_json_safe(v), default=str)))}"
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
    initial_state = _sample_for(case.input, fallback_id=case.id).info.get("initial_state") or {}
    entities = _entity_index(initial_state, end_state if isinstance(end_state, Mapping) else {})
    initial_world = WorldState(**initial_state) if initial_state else None
    held_initially = [
        initial_world is not None and bool(AssertionRegistry.check(initial_world, a)) for a in assertions
    ]
    diffs: _ServiceDiffs | None = None
    if isinstance(end_state, Mapping):
        state: dict[str, Any] = {
            "info": {"assertions": assertions},
            "world": WorldState(**end_state),
            "initial_state": initial_state,
        }
        partial = float(partial_credit(state))
        outcomes: list[Mapping[str, Any]] = list(state.get("_assertion_results") or [])
        diffs = _ServiceDiffs(initial_state, initial_world, end_state)
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
