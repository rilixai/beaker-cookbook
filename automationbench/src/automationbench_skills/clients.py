from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, cast
from uuid import uuid4

import httpx
from automationbench.clients import (
    _NON_RETRYABLE_CHAT,
    OpenAIResponsesClient,
    _is_safety_classifier_error,
    _parse_retry_after,
    _perf,
    _record_model_call,
)
from verifiers.clients import Client, OpenAIChatCompletionsClient
from verifiers.errors import EmptyModelResponseError, ModelError, ToolParseError


DEFAULT_MAX_OUTPUT_TOKENS = 16_384
DEFAULT_MAX_MODEL_ATTEMPTS = 3
DEFAULT_MODEL_REQUEST_TIMEOUT = 300.0
_context: ContextVar[dict[str, Any]] = ContextVar("automationbench_inference_context", default={})
_response_headers: ContextVar[dict[str, Any] | None] = ContextVar("automationbench_response_headers", default=None)


async def _remember_request_id(response: httpx.Response) -> None:
    # Verifiers' Chat sidecar parser drops the SDK's private request ID.
    # Retain only this header, scoped to the current concurrent model turn.
    headers = _response_headers.get()
    if headers is not None:
        headers["request_id"] = response.headers.get("x-request-id")


@contextmanager
def inference_context(case_id: str, task_name: str) -> Iterator[None]:
    token = _context.set({"case_id": case_id, "task_name": task_name, "rollout_id": str(uuid4())})
    try:
        yield
    finally:
        _context.reset(token)


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _usage(response: Any) -> dict[str, int | None]:
    usage = _field(response, "usage")
    return {
        "input_tokens": _field(usage, "prompt_tokens")
        if _field(usage, "prompt_tokens") is not None
        else _field(usage, "input_tokens"),
        "output_tokens": _field(usage, "completion_tokens")
        if _field(usage, "completion_tokens") is not None
        else _field(usage, "output_tokens"),
    }


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def _response_problem(response: Any) -> Exception | None:
    """Validate the entire tool batch before verifiers can execute any of it."""
    choices = _field(response, "choices")
    functions = []
    if choices is not None:
        if not choices:
            return cast(Exception, EmptyModelResponseError("Empty model reply"))
        message = _field(choices[0], "message")
        if not any(_field(message, key) for key in ("content", "tool_calls", "reasoning_content", "reasoning_items")):
            return cast(Exception, EmptyModelResponseError("Empty model reply"))
        functions = [
            _field(tool, "function")
            for choice in choices
            for tool in (_field(_field(choice, "message"), "tool_calls") or [])
        ]
    else:
        output = _field(response, "output")
        if output == []:
            return cast(Exception, EmptyModelResponseError("Empty model reply"))
        functions = [item for item in (output or []) if _field(item, "type") == "function_call"]
    for function in functions:
        if _field(function, "name") != "execute_tool":
            continue
        try:
            outer = json.loads(_field(function, "arguments"), parse_constant=_reject_json_constant)
            nested = outer.get("arguments") if isinstance(outer, dict) else None
            if not isinstance(nested, str) or not isinstance(
                json.loads(nested, parse_constant=_reject_json_constant), dict
            ):
                raise ValueError("Expected a JSON object")
        except (TypeError, ValueError, RecursionError):
            return cast(Exception, ToolParseError("execute_tool.arguments must contain one complete JSON object"))
    return None


class BoundedInferenceClient(Client):
    """One retry layer and one deadline for an entire model turn."""

    def __init__(
        self,
        *args: Any,
        max_attempts: int = DEFAULT_MAX_MODEL_ATTEMPTS,
        request_timeout: float = DEFAULT_MODEL_REQUEST_TIMEOUT,
        **kwargs: Any,
    ) -> None:
        if max_attempts <= 0 or request_timeout <= 0 or not math.isfinite(request_timeout):
            raise ValueError("Model attempts and request timeout must be positive and finite")
        super().__init__(*args, **kwargs)
        self.max_attempts = max_attempts
        self.request_timeout = request_timeout
        # Also covers callers that pass a preconstructed SDK client.
        self._client = self.client.with_options(max_retries=0, timeout=request_timeout)
        hooks = self.client._client.event_hooks["response"]
        if _remember_request_id not in hooks:
            hooks.append(_remember_request_id)

    async def _request(self, *args: Any, **kwargs: Any) -> Any:
        return await super().get_native_response(*args, **kwargs)

    async def get_native_response(
        self, prompt: Any, model: str, sampling_args: Any, tools: Any = None, **kwargs: Any
    ) -> Any:
        started = time.monotonic()
        state = kwargs.get("state")
        effective = {**(sampling_args or {}), **((sampling_args or {}).get("extra_body") or {})}
        cap = effective.get("max_output_tokens", effective.get("max_completion_tokens", effective.get("max_tokens")))
        turn_id = str(uuid4())

        def log(phase: str, attempt: int, **fields: Any) -> None:
            print(
                json.dumps(
                    {
                        "event": "automationbench_model_attempt",
                        **_context.get(),
                        "turn_id": turn_id,
                        "model": model,
                        "max_output_tokens": cap,
                        "attempt": attempt,
                        "phase": phase,
                        "elapsed_ms": round((time.monotonic() - started) * 1000),
                        **fields,
                    }
                ),
                flush=True,
            )

        deadline = asyncio.timeout(self.request_timeout)
        format_retries = 0
        attempt = 0
        headers: dict[str, Any] = {}
        header_token = _response_headers.set(headers)
        try:
            async with deadline:
                for attempt in range(1, self.max_attempts + 1):
                    headers.clear()
                    log("start", attempt)
                    sent = time.monotonic()
                    try:
                        response = await self._request(prompt, model, sampling_args, tools, **kwargs)
                    except Exception as exc:
                        log(
                            "error",
                            attempt,
                            error_type=type(exc).__name__,
                            status_code=getattr(exc, "status_code", None),
                            attempt_elapsed_ms=round((time.monotonic() - sent) * 1000),
                            request_id=getattr(exc, "request_id", None) or headers.get("request_id"),
                        )
                        if _is_safety_classifier_error(exc):
                            raise ModelError("provider safety classifier rejected the request") from exc
                        if isinstance(exc, _NON_RETRYABLE_CHAT) or attempt == self.max_attempts:
                            raise
                        delay = _parse_retry_after(exc)
                        delay = delay if delay is not None else min(2 ** (attempt - 1), 8)
                        log("retry", attempt, retry_delay_seconds=delay)
                        await asyncio.sleep(delay)
                        continue
                    _record_model_call(state, time.monotonic() - sent, response)
                    record_cost(state, response)
                    problem = _response_problem(response)
                    choices = _field(response, "choices")
                    finish = _field(choices[0], "finish_reason") if choices else _field(response, "status")
                    usage = _field(response, "usage")
                    details = _field(usage, "completion_tokens_details") or _field(usage, "output_tokens_details")
                    log(
                        "response",
                        attempt,
                        request_id=getattr(response, "_request_id", None) or headers.get("request_id"),
                        response_id=_field(response, "id"),
                        finish_reason=finish,
                        reasoning_tokens=_field(details, "reasoning_tokens"),
                        attempt_elapsed_ms=round((time.monotonic() - sent) * 1000),
                        **_usage(response),
                        format_valid=problem is None,
                    )
                    if problem is None:
                        return response
                    # The environment counts accepted responses at tool dispatch.
                    # Count rejected replies here, including the final failed one.
                    if state is not None:
                        usage = state.setdefault("_usage", {"input_tokens": 0, "output_tokens": 0})
                        for key, value in _usage(response).items():
                            usage[key] += value or 0
                    if format_retries == 1 or attempt == self.max_attempts:
                        raise problem
                    format_retries += 1
                    log("format_retry", attempt, error_type=type(problem).__name__)
        except TimeoutError:
            if deadline.expired():
                log("deadline", attempt, timeout_seconds=self.request_timeout, request_id=headers.get("request_id"))
            raise
        finally:
            _response_headers.reset(header_token)
        raise RuntimeError("Model attempt budget exhausted")


class CostTrackingChatCompletionsClient(BoundedInferenceClient, OpenAIChatCompletionsClient):
    async def to_native_prompt(self, messages: Any) -> Any:
        # Preserve upstream's compatibility fix for strict gateway validators.
        native, extra = await super().to_native_prompt(messages)
        for message in native:
            if message.get("reasoning_content") is None:
                message.pop("reasoning_content", None)
        return native, extra


class BoundedResponsesClient(BoundedInferenceClient, OpenAIResponsesClient):
    async def _request(self, prompt: Any, model: str, sampling_args: Any, tools: Any = None, **kwargs: Any) -> Any:
        # Reuse request/history conversion, bypassing upstream's 40-attempt loop.
        kwargs.pop("state", None)
        call_kwargs = self.build_call_kwargs(prompt, model, sampling_args, tools, **kwargs)
        return await self.client.responses.create(**call_kwargs)


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
