"""The one place that calls a decision model (and any chat model a candidate adds).

Fixed: not part of the editable surface. Every decision request passes the
request guard (only currently admissible commands as options), is traced as a
``systemone`` model call, and its answer is checked before the agent sees it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Protocol, Self

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    TypeSafeAPIConnectionError,
    TypeSafeAPITimeoutError,
    TypeSafeError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)


# The decision model. Change it here and nowhere else.
MODEL = "jev-1.14.0"
PROVIDER = "typesafe"
SERVER_ADDRESS = "api.typesafe.ai"
REQUEST_TIMEOUT_SECONDS = 120.0
# Reported list price for jev-1.14.0: input tokens only; output tokens are not billed.
PRICE_PER_MILLION_INPUT_TOKENS = 0.042
# The most options one decision request accepts.
OPTION_CAP = 255
QUESTION_KEY = "decision"
# Used only if a candidate adds a chat call and the run selected no model.
CHAT_MODEL = "gpt-6-luna"
CHAT_REASONING_EFFORT = "low"
CHAT_TIMEOUT_SECONDS = 120.0

_RETRYABLE = (TypeSafeRateLimitError, TypeSafeInternalServerError, TypeSafeAPIConnectionError, TypeSafeAPITimeoutError)


class RequestRejected(ValueError):
    """A decision request with an option that is not a currently admissible command."""


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
    """The chosen option and the probability of every option offered."""

    choice: str
    probabilities: dict[str, float]
    model: str
    input_tokens: int | None
    output_tokens: int | None

    @property
    def top_probability(self) -> float:
        return max(self.probabilities.values())


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

    def stage(self, name: str, *, inputs: Any = None) -> AbstractContextManager[Any]: ...


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

    @contextmanager
    def stage(self, name: str, *, inputs: Any = None) -> Iterator[_NoOperation]:
        del name, inputs
        yield _NoOperation()


def guard(criteria: Mapping[str, str], admissible: Sequence[str]) -> None:
    """Reject requests that offer anything but currently admissible commands.

    Keys must be admissible commands, unique by construction, at least two and
    at most ``OPTION_CAP``. Leaving commands out is allowed. Descriptions are
    the agent's to write but must be non-empty text.
    """
    if not isinstance(criteria, Mapping) or len(criteria) < 2:
        raise RequestRejected("A decision request needs at least two options.")
    if len(criteria) > OPTION_CAP:
        raise RequestRejected(f"{len(criteria)} options > {OPTION_CAP}, the most one request accepts.")
    allowed = set(admissible)
    invented = [key for key in criteria if key not in allowed]
    if invented:
        raise RequestRejected(f"Options must be currently admissible commands; not admissible: {invented[:5]}")
    if any(not isinstance(text, str) or not text.strip() for text in criteria.values()):
        raise RequestRejected("Every option needs a non-empty text description.")


@dataclass
class CallRecord:
    operation: str
    model: str
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    top_probability: float | None = None


class Session:
    """What the agent may use during one episode: guarded decision calls and traced chat calls."""

    def __init__(
        self,
        *,
        target: DecisionTarget = DEFAULT_TARGET,
        trace: Tracer | None = None,
        chat_factory: Callable[[], ChatModel] | None = None,
        client: AsyncTypeSafeClient | None = None,
    ) -> None:
        self.target = target
        self.trace: Tracer = trace or NullTracer()
        self.calls: list[CallRecord] = []
        self.admissible: tuple[str, ...] = ()
        self.step_decisions: list[dict[str, Any]] = []
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

    def begin_step(self, admissible: Sequence[str]) -> None:
        """Called by the episode loop before each action; sets what requests may offer."""
        self.admissible = tuple(admissible)
        self.step_decisions = []

    def _decision_client(self) -> AsyncTypeSafeClient:
        if self._client is None:
            self._client = AsyncTypeSafeClient(
                api_key=self.target.api_key, base_url=self.target.base_url, timeout=REQUEST_TIMEOUT_SECONDS
            )
        return self._client

    async def decide(self, *, state: str, instructions: str, criteria: Mapping[str, str]) -> Decision:
        """Ask one native Choice question whose options are admissible commands."""
        try:
            guard(criteria, self.admissible)
        except RequestRejected as exc:
            raise DecisionError("request_rejected", str(exc)) from exc
        if not state.strip():
            raise DecisionError("request_rejected", "The state cannot be empty.")
        question = Choice(instructions=instructions or None, criteria=dict(criteria))
        start = time.monotonic()
        with self.trace.model_call(
            operation="systemone",
            provider=self.target.provider,
            model=self.target.model,
            server_address=self.target.server_address,
            input_messages={"state": state, "questions": {QUESTION_KEY: question.model_dump(mode="json")}},
        ) as call:
            try:
                response = await self._decision_client().system_one(
                    state=state, questions={QUESTION_KEY: question}, model=self.target.model
                )
            except TypeSafeError as exc:
                call.output({"error": type(exc).__name__})
                raise DecisionError(
                    "request_error", f"{type(exc).__name__}: {exc}"[:300], retryable=isinstance(exc, _RETRYABLE)
                ) from exc
            latency = time.monotonic() - start
            usage = response.usage
            call.usage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
            try:
                if self.target.expected_model and response.model != self.target.expected_model:
                    raise ValueError(f"model_version_changed: {response.model}")
                answer = response.choices[QUESTION_KEY]
                probabilities = {key: float(answer.probabilities[key]) for key in criteria}
                if answer.choice not in criteria:
                    raise ValueError(f"choice {answer.choice!r} was not offered")
            except (ValueError, KeyError, TypeError) as exc:
                call.output({"error": f"malformed_answer: {exc}"[:300]})
                raise DecisionError("malformed_answer", str(exc)[:300]) from exc
            call.output({"choice": answer.choice, "probabilities": probabilities})
        decision = Decision(answer.choice, probabilities, response.model, usage.input_tokens, usage.output_tokens)
        self.calls.append(
            CallRecord(
                "systemone", response.model, latency, usage.input_tokens, usage.output_tokens, decision.top_probability
            )
        )
        self.step_decisions.append({"choice": decision.choice, "probabilities": probabilities})
        return decision

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
