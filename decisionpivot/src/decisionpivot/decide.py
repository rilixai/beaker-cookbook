"""The one place that calls a decision model (and any chat model a candidate adds).

Fixed: not part of the editable surface. Every decision request passes the
request guard and is traced; the answer is mapped back to the item's original
option keys and order.
"""

from __future__ import annotations

import math
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Protocol, Self, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import JsonValue
from typesafe_sdk import (
    AsyncTypeSafeClient,
    TypeSafeAPIConnectionError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    TypeSafeError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)

from decisionpivot.data import Item, Question


# The decision model. Change it here and nowhere else.
MODEL = "jev-1.14.0"
PROVIDER = "typesafe"
SERVER_ADDRESS = "api.typesafe.ai"
REQUEST_TIMEOUT_SECONDS = 360.0
# Reported list price for jev-1.14.0: input tokens only; output tokens are not billed.
PRICE_PER_MILLION_INPUT_TOKENS = 0.042
# Used only if a candidate adds a chat call and the run selected no model.
CHAT_MODEL = "gpt-6-luna"
CHAT_REASONING_EFFORT = "low"
CHAT_TIMEOUT_SECONDS = 120.0

_RETRYABLE = (TypeSafeRateLimitError, TypeSafeInternalServerError, TypeSafeAPIConnectionError, TypeSafeAPITimeoutError)


class RequestRejected(ValueError):
    """A request that drops, reorders, adds or edits an option, or drops the question."""


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
class Request:
    """One decision request: a state and one typed question. Build it from the item, then add to it."""

    state: str | dict[str, JsonValue]
    question: Question

    @classmethod
    def of(cls, item: Item) -> Request:
        return cls(state=item.state, question=item.question)


@dataclass(frozen=True)
class Decision:
    """Probabilities keyed by option, in the item's original option order."""

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


def guard(original: Question, request: Question) -> None:
    """Reject requests that could game the benchmark.

    The request keeps the question type and every original option key, in the
    original order; no option is added. Text may be added to the instructions
    and to each option's text, but the original text stays in it.
    """
    if request.type != original.type:
        raise RequestRejected("The question type is fixed.")
    if request.keys != original.keys:
        raise RequestRejected("Every request must carry exactly the original options, in their original order.")
    if original.instructions not in request.instructions:
        raise RequestRejected("The request must keep the original question text.")
    if any(text not in request.criteria[key] for key, text in original.criteria.items()):
        raise RequestRejected("Option texts may be added to, not edited.")


def probabilities_from(answer: Any, keys: Sequence[str]) -> list[float]:
    """An answer object's probabilities over the option keys, normalised as upstream's scorer reads them."""
    if not isinstance(answer, Mapping) or answer.get("type") != "choice":
        raise ValueError("expected a choice answer")
    p = answer.get("probabilities")
    if not isinstance(p, Mapping) or set(p) != set(keys):
        raise ValueError("the answer must give a probability for every option")
    values = [p[k] for k in keys]
    if any(isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("probabilities must be finite and non-negative")
    total = math.fsum(values)
    if total <= 0:
        raise ValueError("probabilities sum to zero")
    return [x / total for x in values]


@dataclass
class CallRecord:
    operation: str
    model: str
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class Session:
    """What the agent may use for one item: guarded decision calls and traced chat calls."""

    def __init__(
        self,
        item: Item,
        *,
        target: DecisionTarget = DEFAULT_TARGET,
        trace: Tracer | None = None,
        chat_factory: Callable[[], ChatModel] | None = None,
        client: AsyncTypeSafeClient | None = None,
    ) -> None:
        self.item = item
        self.target = target
        self.trace: Tracer = trace or NullTracer()
        self.calls: list[CallRecord] = []
        self.decisions: list[dict[str, float]] = []
        self._chat_factory = chat_factory
        self._chat: ChatModel | None = None
        self._client = client
        self._owns_client = client is None

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

    async def decide(self, request: Request) -> Decision:
        original = self.item.question
        guard(original, request.question)
        question = request.question.model_dump(mode="json")
        start = time.monotonic()
        with self.trace.model_call(
            operation="systemone",
            provider=self.target.provider,
            model=self.target.model,
            server_address=self.target.server_address,
            input_messages={"state": request.state, "questions": {"q": question}},
        ) as call:
            try:
                response = await self._decision_client().system_one(
                    state=request.state,
                    # Sent as a raw dict: the SDK passes it on unchanged, keeping upstream's key order.
                    questions={"q": cast(Any, question)},
                    model=self.target.model,
                )
            except TypeSafeAPIResponseValidationError as exc:
                call.output({"error": "malformed_answer"})
                raise DecisionError("malformed_answer", "the reply does not match the API schema") from exc
            except TypeSafeError as exc:
                call.output({"error": type(exc).__name__})
                raise DecisionError(
                    "request_error", type(exc).__name__, retryable=isinstance(exc, _RETRYABLE)
                ) from exc
            finally:
                latency = time.monotonic() - start
            usage = response.usage
            call.usage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
            answer = response.raw_http_response.json().get("answers", {}).get("q")
            try:
                if self.target.expected_model and response.model != self.target.expected_model:
                    raise ValueError(f"model_version_changed: {response.model}")
                p = probabilities_from(answer, original.keys)
            except ValueError as exc:
                call.output({"error": f"malformed_answer: {exc}"[:300]})
                raise DecisionError("malformed_answer", str(exc)[:300]) from exc
            call.output(answer)
        self.calls.append(CallRecord("systemone", response.model, latency, usage.input_tokens, usage.output_tokens))
        self.decisions.append(dict(zip(original.keys, p, strict=True)))
        return Decision(
            probabilities=self.decisions[-1],
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


Solver = Callable[[Item, Session, str], Awaitable[Mapping[str, float]]]


@dataclass
class Outcome:
    item_id: str
    probabilities: dict[str, float] | None
    error: str | None
    retryable: bool
    calls: list[CallRecord]
    latency_s: float

    def record(self) -> dict[str, Any]:
        decisions = [c for c in self.calls if c.operation == "systemone"]
        return {
            "id": self.item_id,
            "probabilities": self.probabilities,
            "error": self.error,
            "decision_calls": len(decisions),
            "chat_calls": len(self.calls) - len(decisions),
            "models": sorted({c.model for c in self.calls}),
            "decision_input_tokens": sum(c.input_tokens or 0 for c in decisions),
            "decision_output_tokens": sum(c.output_tokens or 0 for c in decisions),
            "decision_latency_s": [c.latency_s for c in decisions],
            "latency_s": self.latency_s,
        }


def validate_answer(
    item: Item, answer: Mapping[str, float], decisions: Sequence[Mapping[str, float]]
) -> dict[str, float]:
    """The final distribution covers exactly the item's options and is one the decision model returned.

    The agent may make several decision calls and pick among them, and may use chat
    calls to shape its requests, but the decision itself comes from the decision model.
    """
    keys = item.question.keys
    if not isinstance(answer, Mapping) or set(answer) != set(keys):
        raise DecisionError("missing_option", "the answer must give a probability for every original option")
    try:
        p = dict(zip(keys, probabilities_from({"type": "choice", "probabilities": dict(answer)}, keys), strict=True))
    except ValueError as exc:
        raise DecisionError("malformed_answer", str(exc)) from exc
    if not any(all(math.isclose(p[k], d[k], rel_tol=1e-9, abs_tol=1e-12) for k in keys) for d in decisions):
        raise DecisionError(
            "answer_not_from_decision", "the answer must be the probabilities of one of this item's decision calls"
        )
    return p


async def run_item(
    item: Item,
    *,
    solver: Solver,
    instructions: str,
    target: DecisionTarget = DEFAULT_TARGET,
    trace: Tracer | None = None,
    chat_factory: Callable[[], ChatModel] | None = None,
    client: AsyncTypeSafeClient | None = None,
) -> Outcome:
    """Run the agent on one item, alone. Every failure is returned with an observable error."""
    start = time.monotonic()
    probabilities: dict[str, float] | None = None
    error: str | None = None
    retryable = False
    async with Session(item, target=target, trace=trace, chat_factory=chat_factory, client=client) as session:
        try:
            answer = await solver(item, session, instructions)
            probabilities = validate_answer(item, answer, session.decisions)
        except DecisionError as exc:
            error, retryable = str(exc), exc.retryable
        except Exception as exc:  # a candidate bug is a failed item, not a crashed run
            error = f"{type(exc).__name__}: {exc}"[:500]
    return Outcome(item.id, probabilities, error, retryable, session.calls, time.monotonic() - start)
