"""The one place that calls a decision model (and any chat model a candidate adds).

Fixed: not part of the editable surface. Every request passes the request
guard, is built by upstream ``mmdm_eval.common.wire``, is traced, and its
answer is aligned to option order by upstream ``mmdm_eval.common.parse_native``.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self
from urllib.request import urlopen

from mmdm_eval.common import parse_native, valid_p, wire
from mmdm_eval.context import PROTOCOL as TRUNCATION_PROTOCOL
from mmdm_eval.context import Budget, FixedFieldsTooLong, adapt_state
from mmdm_eval.runner import request_counter
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import TypeAdapter
from typesafe_sdk import (
    AsyncTypeSafeClient,
    TypeSafeAPIConnectionError,
    TypeSafeAPITimeoutError,
    TypeSafeError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)
from typesafe_sdk import (
    Question as ApiQuestion,
)

from mmdm.data import OPTION_CAP, Question


# The decision model. Change it here and nowhere else.
MODEL = "jev-1.14.0"
PROVIDER = "typesafe"
SERVER_ADDRESS = "api.typesafe.ai"
# Upstream examples/evaluate_jev.sh: --context-limit 32768 --service-overhead 1024.
CONTEXT_LIMIT = 32_768
SERVICE_OVERHEAD = 1_024
BUDGET = Budget(CONTEXT_LIMIT, SERVICE_OVERHEAD)
REQUEST_TIMEOUT_SECONDS = 360.0
# Reported list price for jev-1.14.0: input tokens only; output tokens are not billed.
PRICE_PER_MILLION_INPUT_TOKENS = 0.042
# Upstream's proxy tokenizer for request budgets, pinned.
TOKENIZER_REPO = "Qwen/Qwen3.5-4B"
TOKENIZER_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
TOKENIZER_URL = f"https://huggingface.co/{TOKENIZER_REPO}/resolve/{TOKENIZER_REVISION}/tokenizer.json"
TOKENIZER_CACHE = Path.home() / ".cache/mmdm" / f"tokenizer-{TOKENIZER_REVISION[:12]}"
# Used only if a candidate adds a chat call and the run selected no model.
CHAT_MODEL = "gpt-6-luna"
CHAT_REASONING_EFFORT = "low"
CHAT_TIMEOUT_SECONDS = 120.0

_API_QUESTION: TypeAdapter[Any] = TypeAdapter(ApiQuestion)
_RETRYABLE = (TypeSafeRateLimitError, TypeSafeInternalServerError, TypeSafeAPIConnectionError, TypeSafeAPITimeoutError)


class RequestRejected(ValueError):
    """A request that changes, drops or adds an option, or edits fixed fields."""


class DecisionError(RuntimeError):
    def __init__(self, status: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(f"{status}: {message}")
        self.status = status
        self.retryable = retryable


@dataclass(frozen=True)
class DecisionTarget:
    """Where decision calls go. The default is native Jev; a run's model choice replaces it."""

    model: str
    provider: str
    server_address: str
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    expected_model: str | None = None


DEFAULT_TARGET = DecisionTarget(model=MODEL, provider=PROVIDER, server_address=SERVER_ADDRESS, expected_model=MODEL)


@dataclass(frozen=True)
class Decision:
    """Probabilities keyed by option ID, in the original option order."""

    probabilities: dict[str, float]
    model: str
    input_tokens: int | None
    output_tokens: int | None


@dataclass
class ChatModel:
    client: AsyncOpenAI
    provider: str
    model: str
    reasoning_effort: str | None = None


class TraceOperation(Protocol):
    def output(self, value: Any) -> None: ...
    def usage(self, *, input_tokens: int | None = None, output_tokens: int | None = None) -> None: ...


class Tracer(Protocol):
    def model_call(
        self, *, operation: str, provider: str, model: str, input_messages: Any, server_address: str | None = None
    ) -> AbstractContextManager[Any]: ...


class _NoOperation:
    def output(self, value: Any) -> None:
        del value

    def usage(self, *, input_tokens: int | None = None, output_tokens: int | None = None) -> None:
        del input_tokens, output_tokens


class NullTracer:
    """For local evaluation without Beaker."""

    @contextmanager
    def model_call(
        self, *, operation: str, provider: str, model: str, input_messages: Any, server_address: str | None = None
    ) -> Iterator[_NoOperation]:
        del operation, provider, model, input_messages, server_address
        yield _NoOperation()


def actor(question: Question) -> dict[str, Any]:
    """Upstream's actor view of a question: state, question, options, task, score_values."""
    return question.model_dump(mode="json")


def guard(original: Question, request: Question) -> None:
    """Reject requests that could game the benchmark.

    Every original option must be present, unchanged (ID, name, criteria), and
    no option may be added. Task and score values are fixed. The request's
    instructions must contain the original question; text may be added around
    it. The state is the candidate's to add to or pre-process. Options may be
    reordered; answers are mapped back by option ID.
    """
    if request.task != original.task or request.score_values != original.score_values:
        raise RequestRejected("The task type and score values are fixed.")
    if request.task == "score" and request.options != original.options:
        raise RequestRejected("Score rubric levels must keep their original order.")
    originals = {o.id: o for o in original.options}
    requested = {o.id: o for o in request.options}
    if len(request.options) != len(original.options) or requested.keys() != originals.keys():
        raise RequestRejected("Every request must contain exactly the original options; none dropped or added.")
    if any(requested[oid] != option for oid, option in originals.items()):
        raise RequestRejected("Option names and criteria cannot be edited.")
    if original.question not in request.question:
        raise RequestRejected("The request must keep the original question text.")


def tokenizer_path(cache_dir: Path = TOKENIZER_CACHE) -> Path:
    """The pinned proxy tokenizer, fetched once and checksum-verified."""
    path = cache_dir / "tokenizer.json"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        with urlopen(TOKENIZER_URL, timeout=300) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != TOKENIZER_SHA256:
            raise ValueError("Tokenizer checksum mismatch")
        partial = path.with_suffix(".partial")
        partial.write_bytes(content)
        partial.replace(path)
    elif hashlib.sha256(path.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("Cached tokenizer checksum mismatch")
    return path


@cache
def default_counter(model: str) -> Any:
    """Upstream's request counter: compact wire JSON measured with the proxy tokenizer."""
    return request_counter(tokenizer_path(), model, "proxy")


def _wire_bytes(value: Mapping[str, Any], model: str) -> int:
    body = wire(dict(value), model)[0]
    return len(json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())


def request_tokens(question: Question, *, model: str = MODEL, counter: Any = None) -> int | None:
    """Token count of the full request, or None when it plainly fits.

    A byte-level BPE token covers at least one byte, so a request whose wire
    JSON is within the token budget in bytes cannot exceed it in tokens; such
    requests skip the tokenizer.
    """
    value = actor(question)
    if counter is None and _wire_bytes(value, model) <= BUDGET.available:
        return None
    return int((counter or default_counter(model))(value))


def head_tail(question: Question, *, model: str = MODEL, counter: Any = None) -> tuple[Question, dict[str, Any]]:
    """Upstream ``mmdm-state-head-tail-v1`` truncation of the state at the upstream Jev budget."""
    value = actor(question)
    if counter is None and _wire_bytes(value, model) <= BUDGET.available:
        return question, {"protocol": TRUNCATION_PROTOCOL, "truncated": False, "status": "fits_by_bytes"}
    try:
        view, meta = adapt_state(value, counter or default_counter(model), BUDGET)
    except FixedFieldsTooLong as exc:
        raise DecisionError("fixed_fields_exceed_budget", "question, options and overhead exceed the budget") from exc
    return question.model_copy(update={"state": view["state"]}), meta


@dataclass
class CallRecord:
    operation: str
    model: str
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class Session:
    """What the agent may use for one question: guarded decision calls and traced chat calls."""

    def __init__(
        self,
        question: Question,
        *,
        target: DecisionTarget = DEFAULT_TARGET,
        trace: Tracer | None = None,
        chat_factory: Callable[[], ChatModel] | None = None,
        client: AsyncTypeSafeClient | None = None,
        counter: Any = None,
    ) -> None:
        self.question = question
        self.target = target
        self.trace: Tracer = trace or NullTracer()
        self.calls: list[CallRecord] = []
        self._chat_factory = chat_factory
        self._chat: ChatModel | None = None
        self._client = client
        self._owns_client = client is None
        self.counter = counter  # None: the pinned proxy tokenizer

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        if self._chat is not None:
            await self._chat.client.close()

    def _decision_client(self) -> AsyncTypeSafeClient:
        if self._client is None:
            self._client = AsyncTypeSafeClient(
                api_key=self.target.api_key, base_url=self.target.base_url, timeout=REQUEST_TIMEOUT_SECONDS
            )
        return self._client

    async def decide(self, request: Question) -> Decision:
        guard(self.question, request)
        if len(request.options) > OPTION_CAP:
            # prepare drops these questions; a hand-made row must not reach the API either.
            raise DecisionError("over_option_cap", f"{len(request.options)} options > {OPTION_CAP}")
        tokens = request_tokens(request, model=self.target.model, counter=self.counter)
        if tokens is not None and tokens > BUDGET.available:
            raise DecisionError("context_budget_exceeded", f"{tokens} request tokens > {BUDGET.available} available")
        body, keys = wire(actor(request), self.target.model)
        api_question = _API_QUESTION.validate_python(body["questions"]["decision"])
        start = time.monotonic()
        with self.trace.model_call(
            operation="systemone",
            provider=self.target.provider,
            model=self.target.model,
            server_address=self.target.server_address,
            input_messages={"state": body["state"], "questions": body["questions"]},
        ) as call:
            try:
                response = await self._decision_client().system_one(
                    state=body["state"], questions={"decision": api_question}, model=self.target.model
                )
            except TypeSafeError as exc:
                # str() of an SDK error carries status, server message and request ID, never credentials.
                detail = f"{type(exc).__name__}: {exc}"[:500]
                call.output({"error": detail})
                raise DecisionError("request_error", detail, retryable=isinstance(exc, _RETRYABLE)) from exc
            finally:
                latency = time.monotonic() - start
            usage = response.usage
            call.usage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
            raw = response.raw_http_response.json()
            try:
                if self.target.expected_model and response.model != self.target.expected_model:
                    raise ValueError(f"model_version_changed: {response.model}")
                p = parse_native(body, keys, raw)
            except (ValueError, KeyError, TypeError, AssertionError) as exc:
                call.output({"error": f"malformed_answer: {exc}"[:300]})
                raise DecisionError("malformed_answer", str(exc)[:300]) from exc
            call.output(raw.get("answers", {}).get("decision"))
        self.calls.append(CallRecord("systemone", response.model, latency, usage.input_tokens, usage.output_tokens))
        by_id = {option.id: float(x) for option, x in zip(request.options, p, strict=True)}
        return Decision(
            probabilities={oid: by_id[oid] for oid in self.question.option_ids},
            model=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )

    async def chat(self, messages: Sequence[ChatCompletionMessageParam]) -> str:
        """A traced chat call that follows the run's model choice."""
        if self._chat is None:
            if self._chat_factory is None:
                raise RuntimeError("No chat model is configured for this session.")
            self._chat = self._chat_factory()
        chat = self._chat
        extra: dict[str, Any] = {"reasoning_effort": chat.reasoning_effort} if chat.reasoning_effort else {}
        start = time.monotonic()
        with self.trace.model_call(
            operation="chat", provider=chat.provider, model=chat.model, input_messages=list(messages)
        ) as call:
            completion = await chat.client.chat.completions.create(model=chat.model, messages=messages, **extra)
            text = completion.choices[0].message.content or ""
            usage = completion.usage
            call.output(text)
            call.usage(
                input_tokens=usage.prompt_tokens if usage else None,
                output_tokens=usage.completion_tokens if usage else None,
            )
        self.calls.append(
            CallRecord(
                "chat",
                chat.model,
                time.monotonic() - start,
                usage.prompt_tokens if usage else None,
                usage.completion_tokens if usage else None,
            )
        )
        return text


def default_chat_model() -> ChatModel:
    return ChatModel(
        client=AsyncOpenAI(timeout=CHAT_TIMEOUT_SECONDS),
        provider="openai",
        model=CHAT_MODEL,
        reasoning_effort=CHAT_REASONING_EFFORT,
    )


Solver = Callable[[Question, Session, str], Awaitable[Mapping[str, float]]]


@dataclass
class Outcome:
    probabilities: dict[str, float] | None
    error: str | None
    retryable: bool
    calls: list[CallRecord]
    latency_s: float

    def record(self) -> dict[str, Any]:
        decisions = [c for c in self.calls if c.operation == "systemone"]
        return {
            "probabilities": self.probabilities,
            "error": self.error,
            "mode": "native",
            "decision_calls": len(decisions),
            "chat_calls": len(self.calls) - len(decisions),
            "models": sorted({c.model for c in self.calls}),
            "decision_input_tokens": sum(c.input_tokens or 0 for c in decisions),
            "decision_output_tokens": sum(c.output_tokens or 0 for c in decisions),
            "latency_s": self.latency_s,
        }


def validate_answer(question: Question, answer: Mapping[str, float]) -> dict[str, float]:
    """The agent's final distribution must cover exactly the original options."""
    if not isinstance(answer, Mapping) or set(answer) != set(question.option_ids):
        raise DecisionError("missing_option", "the answer must give a probability for every original option")
    values = [answer[oid] for oid in question.option_ids]
    if any(isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(x) for x in values):
        raise DecisionError("malformed_answer", "probabilities must be finite numbers")
    try:
        normalized = valid_p(values, len(values))
    except AssertionError as exc:
        raise DecisionError("malformed_answer", "probabilities must be in [0, 1] and sum to one") from exc
    return dict(zip(question.option_ids, normalized, strict=True))


async def run_question(
    question: Question,
    *,
    solver: Solver,
    instructions: str,
    target: DecisionTarget = DEFAULT_TARGET,
    trace: Tracer | None = None,
    chat_factory: Callable[[], ChatModel] | None = None,
    client: AsyncTypeSafeClient | None = None,
    counter: Any = None,
) -> Outcome:
    """Run the agent on one question. Every failure is returned with an observable error."""
    start = time.monotonic()
    probabilities: dict[str, float] | None = None
    error: str | None = None
    retryable = False
    async with Session(
        question, target=target, trace=trace, chat_factory=chat_factory, client=client, counter=counter
    ) as session:
        try:
            probabilities = validate_answer(question, await solver(question, session, instructions))
        except DecisionError as exc:
            error, retryable = str(exc), exc.retryable
        except Exception as exc:  # a candidate bug is a failed case, not a crashed run
            error = f"{type(exc).__name__}: {exc}"[:500]
    return Outcome(probabilities, error, retryable, session.calls, time.monotonic() - start)
