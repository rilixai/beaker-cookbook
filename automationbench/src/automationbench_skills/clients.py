from __future__ import annotations

import asyncio
import math
import time
from typing import Any

from automationbench.clients import (
    _NON_RETRYABLE_CHAT,
    _is_safety_classifier_error,
    _parse_retry_after,
    _perf,
    _record_model_call,
    _record_safety_classifier_error,
)
from verifiers.clients import OpenAIChatCompletionsClient
from verifiers.clients.openai_chat_completions_client import parse_reasoning_content
from verifiers.errors import ModelError


# Reasoning counts toward the completion cap on most providers, so the cap is a
# reasoning allowance plus room to answer. In hosted runs, answers stayed under
# 3k tokens while reasoning reached about 26k.
DEFAULT_MAX_REASONING_TOKENS = 24_576
DEFAULT_MAX_ANSWER_TOKENS = 8_192
DEFAULT_MAX_MODEL_ATTEMPTS = 3
# Each attempt times out on its own, and the whole turn, retries and backoff
# included, has a longer deadline. The hosted gateway gives up on a non-streamed
# reply at 280 seconds, so a retry still fits after it.
DEFAULT_MODEL_REQUEST_TIMEOUT = 300.0
DEFAULT_MODEL_TURN_TIMEOUT = 600.0


class CostTrackingChatCompletionsClient(OpenAIChatCompletionsClient):
    """Chat requests with one retry layer and a deadline for the whole model turn."""

    def __init__(
        self,
        *args: Any,
        max_attempts: int = DEFAULT_MAX_MODEL_ATTEMPTS,
        request_timeout: float = DEFAULT_MODEL_REQUEST_TIMEOUT,
        turn_timeout: float = DEFAULT_MODEL_TURN_TIMEOUT,
        **kwargs: Any,
    ) -> None:
        if max_attempts <= 0 or any(t <= 0 or not math.isfinite(t) for t in (request_timeout, turn_timeout)):
            raise ValueError("Model attempts and timeouts must be positive and finite")
        super().__init__(*args, **kwargs)
        self.max_attempts = max_attempts
        self.request_timeout = request_timeout
        self.turn_timeout = turn_timeout
        self._client = self.client.with_options(max_retries=0, timeout=request_timeout)

    async def to_native_prompt(self, messages: Any) -> Any:
        # Preserve upstream's compatibility fix for strict gateway validators.
        native, extra = await super().to_native_prompt(messages)
        for message in native:
            if message.get("reasoning_content") is None:
                message.pop("reasoning_content", None)
        return native, extra

    async def get_native_response(self, *args: Any, **kwargs: Any) -> Any:
        state = kwargs.get("state")
        async with asyncio.timeout(self.turn_timeout):
            for attempt in range(self.max_attempts):
                try:
                    started = time.monotonic()
                    response = await super().get_native_response(*args, **kwargs)
                except Exception as exc:
                    if _is_safety_classifier_error(exc):
                        _record_safety_classifier_error(state, exc)
                        raise ModelError("provider safety classifier rejected the request") from exc
                    if isinstance(exc, _NON_RETRYABLE_CHAT) or attempt == self.max_attempts - 1:
                        raise
                    retry_after = _parse_retry_after(exc)
                    await asyncio.sleep(retry_after if retry_after is not None else 2**attempt)
                    continue
                _record_model_call(state, time.monotonic() - started, response)
                record_cost(state, response)
                # Keep upstream's empty-response retry. Tool arguments pass through
                # unchanged to the environment's existing parser and execution loop.
                choices = getattr(response, "choices", None)
                if choices:
                    choice = choices[0]
                    message = choice.message
                    if choice.finish_reason == "length" and not (message.content or message.tool_calls):
                        # Reasoning used the whole cap. Retrying the same request
                        # would stop in the same place, and returning it would end
                        # the rollout as if the agent had finished.
                        raise ModelError("the model's reasoning used the whole output cap without answering")
                    if not (message.content or message.tool_calls or parse_reasoning_content(message)):
                        if attempt < self.max_attempts - 1:
                            if state is not None:
                                usage = state.setdefault("_usage", {"input_tokens": 0, "output_tokens": 0})
                                usage["input_tokens"] += response.usage.prompt_tokens if response.usage else 0
                                usage["output_tokens"] += response.usage.completion_tokens if response.usage else 0
                            await asyncio.sleep(2**attempt)
                            continue
                return response
        raise RuntimeError("Model attempt budget exhausted")


def record_cost(state: Any, native_response: Any) -> None:
    usage = getattr(native_response, "usage", None)
    if usage is None:
        return
    if isinstance(usage, dict):
        cost = usage.get("cost")
    else:
        cost = getattr(usage, "cost", None)
        if cost is None:
            model_extra = getattr(usage, "model_extra", None) or {}
            cost = model_extra.get("cost")
    if not isinstance(cost, (int, float)) or isinstance(cost, bool):
        return
    perf = _perf(state)
    if perf is not None:
        perf["cost_usd"] = perf.get("cost_usd", 0.0) + float(cost)
