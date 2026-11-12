"""The agent's model, pinned: the agent picks messages and tools, never the model or its settings."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import openai
from beaker.tracing import Trace
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion


class TransientServiceError(RuntimeError):
    """A provider failed in a way a later attempt of the whole case may not."""


class CallLimitReached(RuntimeError):
    pass


@dataclass
class ChatUsage:
    calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    seconds: float = 0.0


@dataclass
class ChatModel:
    """One case's model access, with a fixed call cap and every call traced and metered."""

    client: AsyncOpenAI
    model: str
    provider: str
    server_address: str
    max_calls: int
    trace: Trace
    request_options: Mapping[str, Any] = field(default_factory=dict)
    usage: ChatUsage = field(default_factory=ChatUsage)

    @property
    def remaining_calls(self) -> int:
        return self.max_calls - self.usage.calls

    async def complete(
        self,
        messages: Iterable[Mapping[str, Any]],
        *,
        tools: Iterable[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
    ) -> ChatCompletion:
        if self.remaining_calls <= 0:
            raise CallLimitReached(f"The agent may make at most {self.max_calls} model calls per question")
        messages = list(messages)
        request: dict[str, Any] = {"model": self.model, "messages": messages, **self.request_options}
        if tools is not None:
            request["tools"] = list(tools)
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        self.usage.calls += 1
        with self.trace.model_call(
            operation="chat",
            provider=self.provider,
            model=self.model,
            server_address=self.server_address,
            input_messages=messages,
        ) as call:
            started = time.perf_counter()
            try:
                completion: ChatCompletion = await self.client.chat.completions.create(**request)
            except (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError) as exc:
                raise TransientServiceError(f"{self.provider} chat completion failed: {type(exc).__name__}") from exc
            finally:
                self.usage.seconds += time.perf_counter() - started
            call.output(completion)
        if completion.usage is not None:
            self.usage.input_tokens += completion.usage.prompt_tokens
            self.usage.output_tokens += completion.usage.completion_tokens
            prompt_details = completion.usage.prompt_tokens_details
            completion_details = completion.usage.completion_tokens_details
            self.usage.cached_input_tokens += (prompt_details.cached_tokens or 0) if prompt_details else 0
            self.usage.reasoning_tokens += (completion_details.reasoning_tokens or 0) if completion_details else 0
        return completion
