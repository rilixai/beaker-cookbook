from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from beaker.tracing import local_capture
from openai import APIStatusError, AsyncOpenAI
from verifiers.errors import ModelError

from automationbench_skills.clients import CostTrackingChatCompletionsClient
from automationbench_skills.runner import ModelSpec


def _reply(*, empty: bool = False) -> dict[str, Any]:
    return {
        "id": "reply",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5.6-luna",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": None if empty else "done"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18, "cost": 0.01},
    }


def _client(handler: Any, **kwargs: Any) -> CostTrackingChatCompletionsClient:
    return CostTrackingChatCompletionsClient(
        AsyncOpenAI(
            api_key="test", max_retries=9, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        ),
        **kwargs,
    )


@pytest.mark.parametrize("extra_body", [None, '{"max_tokens":128000}', '{"max_output_tokens":1024}'])
async def test_output_cap_reaches_the_wire(extra_body: str | None) -> None:
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_reply())

    model = ModelSpec(api="chat_completions", extra_body=extra_body)
    client = _client(handler)
    try:
        await client.get_native_response([], model.name, model.sampling_args())
    finally:
        await client.close()
    (body,) = captured
    assert body["max_completion_tokens"] == (1024 if extra_body and "1024" in extra_body else 32768)
    assert "max_tokens" not in body
    assert "max_output_tokens" not in body
    assert "parallel_tool_calls" not in body


@pytest.mark.parametrize(("status", "attempts"), [(503, 3), (429, 3), (400, 1), (401, 1), (403, 1)])
async def test_retries_are_bounded_and_client_errors_fail_fast(status: int, attempts: int) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status, json={"error": {"message": "failed", "type": "api_error"}}, headers={"retry-after": "0"}
        )

    client = _client(handler)
    assert client.client.max_retries == 0
    assert client.client.timeout == 300
    try:
        with pytest.raises(APIStatusError):
            await client.get_native_response([], "gpt-5.6-luna", {})
    finally:
        await client.close()
    assert len(requests) == attempts


@pytest.mark.parametrize("backoff", [False, True])
async def test_deadline_cancels_a_stalled_request_or_backoff(backoff: bool) -> None:
    requests = []
    cancelled = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if backoff:
            return httpx.Response(429, json={"error": {"message": "wait"}}, headers={"retry-after": "60"})
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.set()
        return httpx.Response(200, json=_reply())

    client = _client(handler, turn_timeout=0.03)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            await client.get_native_response([], "gpt-5.6-luna", {})
    finally:
        await client.close()
    assert time.monotonic() - started < 1
    assert len(requests) == 1
    assert backoff or cancelled.is_set()


async def test_empty_response_retry_retains_usage_and_cost() -> None:
    replies = [_reply(empty=True), _reply()]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=replies.pop(0))

    client = _client(handler)
    state: dict[str, Any] = {}
    try:
        response = await client.get_native_response([], "gpt-5.6-luna", {}, state=state)
    finally:
        await client.close()
    assert response.choices[0].message.content == "done"
    assert state["_usage"] == {"input_tokens": 11, "output_tokens": 7}
    assert state["_perf"]["model_calls"] == 2
    assert state["_perf"]["cost_usd"] == pytest.approx(0.02)


async def test_safety_rejection_is_not_retried_and_remains_visible() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            400, json={"error": {"message": "ContentPolicyViolationError", "type": "invalid_prompt"}}
        )

    client = _client(handler)
    state: dict[str, Any] = {}
    try:
        with pytest.raises(ModelError):
            await client.get_native_response([], "gpt-5.6-luna", {}, state=state)
    finally:
        await client.close()
    assert len(requests) == 1
    assert state["_debug"]["errors"][0]["type"] == "safety_classifier"


async def test_hosted_trace_preserves_retry_and_malformed_tool_response(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).parents[1] / ".beaker"))
    from beaker_integration import _TracedChatCompletionsClient

    reply = _reply()
    arguments = [json.dumps({"tool_name": "example", "arguments": value}) for value in ["{}", "{}garbage"]]
    reply["choices"][0]["message"] = {
        "role": "assistant",
        "tool_calls": [
            {"id": f"call_{i}", "type": "function", "function": {"name": "execute_tool", "arguments": value}}
            for i, value in enumerate(arguments)
        ],
    }
    reply["choices"][0]["finish_reason"] = "tool_calls"
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503, json={"error": {"message": "retry"}}, headers={"retry-after": "0"})
        return httpx.Response(200, json=reply)

    client = _TracedChatCompletionsClient(
        AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    )
    state: dict[str, Any] = {}
    try:
        with local_capture(tmp_path, case_id="case", candidate_id="candidate", strict_evidence=False) as capture:
            response = await client.get_native_response([], "gpt-5.6-luna", ModelSpec().sampling_args(), state=state)
    finally:
        await client.close()
    assert len(requests) == 2
    assert [call.function.arguments for call in response.choices[0].message.tool_calls] == arguments
    assert state["_perf"]["cost_usd"] == pytest.approx(0.01)
    assert capture.receipt is not None
    calls = capture.receipt.to_dict()["projection"]["model_calls"]
    assert len(calls) == 2
    assert calls[0]["output"] is None
    assert calls[1]["usage"]["output_tokens"] == 7
    batches = [
        json.loads(line) for path in tmp_path.glob("captures/*.otlp.jsonl") for line in path.read_text().splitlines()
    ]
    spans = [
        span
        for batch in batches
        for resource in batch["resourceSpans"]
        for scope in resource["scopeSpans"]
        for span in scope["spans"]
    ]
    assert any(span.get("status", {}).get("code") == 2 and "503" in span["status"]["message"] for span in spans)


@pytest.mark.parametrize("selected_model", [False, True])
async def test_hosted_client_creation_uses_the_limits(monkeypatch: Any, selected_model: bool) -> None:
    sys.path.insert(0, str(Path(__file__).parents[1] / ".beaker"))
    import beaker_integration

    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_reply())

    def sdk(**kwargs: Any) -> AsyncOpenAI:
        return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(beaker_integration, "AsyncOpenAI", sdk)
    monkeypatch.setattr(
        beaker_integration,
        "inference_target",
        lambda runtime: SimpleNamespace(model="gpt-5.6-luna", api_key="test", base_url="https://gateway.invalid/v1"),
    )
    client, model = beaker_integration._client_for(SimpleNamespace(model=object() if selected_model else None))
    try:
        assert client.client.max_retries == 0
        assert client.client.timeout == 300
        assert client.turn_timeout == 600
        await client.get_native_response([], model.name, model.sampling_args())
    finally:
        await client.close()
    assert captured[0]["max_completion_tokens"] == 32768


def test_native_clients_keep_their_existing_sampling_settings() -> None:
    assert "max_completion_tokens" not in ModelSpec(api="responses").sampling_args()
    assert "max_completion_tokens" not in ModelSpec(name="claude-sonnet-5-5", api="anthropic").sampling_args()


@pytest.mark.parametrize("field", ["request_timeout", "turn_timeout"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_deadline_cannot_disable_the_guard(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        CostTrackingChatCompletionsClient(None, **{field: value})


@pytest.mark.parametrize(
    ("extra_body", "cap"),
    [
        (None, 32768),
        ('{"reasoning": {"max_tokens": 12000}}', 20192),
        ('{"reasoning": {"max_tokens": 12000}, "max_tokens": 4096}', 4096),
    ],
)
def test_a_reasoning_budget_sets_the_cap_with_room_to_answer(extra_body: str | None, cap: int) -> None:
    args = ModelSpec(api="chat_completions", extra_body=extra_body).sampling_args()
    assert args["max_completion_tokens"] == cap
    if extra_body and "reasoning" in extra_body:
        assert args["extra_body"]["reasoning"] == {"max_tokens": 12000}


@pytest.mark.parametrize("budget", ["0", "-1", "true", '"12000"'])
def test_an_invalid_reasoning_budget_is_rejected(budget: str) -> None:
    with pytest.raises(ValueError, match="reasoning.max_tokens"):
        ModelSpec(api="chat_completions", extra_body=f'{{"reasoning": {{"max_tokens": {budget}}}}}').sampling_args()


def _stopped_at_the_cap(content: str | None) -> dict[str, Any]:
    reply = _reply()
    reply["choices"][0]["message"] = {"role": "assistant", "content": content, "reasoning_content": "thinking"}
    reply["choices"][0]["finish_reason"] = "length"
    return reply


async def test_reasoning_that_fills_the_cap_fails_without_a_retry() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_stopped_at_the_cap(None))

    client = _client(handler)
    state: dict[str, Any] = {}
    try:
        with pytest.raises(ModelError, match="whole output cap"):
            await client.get_native_response([], "gpt-5.6-luna", {}, state=state)
    finally:
        await client.close()
    assert len(requests) == 1
    # The call was paid for, so its cost stays accounted.
    assert state["_perf"]["model_calls"] == 1
    assert state["_perf"]["cost_usd"] == pytest.approx(0.01)


async def test_an_answer_cut_off_at_the_cap_is_returned() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_stopped_at_the_cap("partial answer"))

    client = _client(handler)
    try:
        response = await client.get_native_response([], "gpt-5.6-luna", {})
    finally:
        await client.close()
    assert (response.choices[0].finish_reason, response.choices[0].message.content) == ("length", "partial answer")


@pytest.mark.parametrize(("turn_timeout", "recovers"), [(1.0, True), (0.35, False)])
async def test_a_retry_after_a_slow_failure_needs_room_in_the_turn(turn_timeout: float, recovers: bool) -> None:
    # Scaled down: a first attempt that fails late, as at the gateway's 280-second
    # timeout, then a retry that takes a while too.
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            await asyncio.sleep(0.3)
            return httpx.Response(503, json={"error": {"message": "timed out"}}, headers={"retry-after": "0"})
        await asyncio.sleep(0.2)
        return httpx.Response(200, json=_reply())

    client = _client(handler, turn_timeout=turn_timeout)
    try:
        if recovers:
            response = await client.get_native_response([], "gpt-5.6-luna", {})
            assert response.choices[0].message.content == "done"
        else:
            with pytest.raises(TimeoutError):
                await client.get_native_response([], "gpt-5.6-luna", {})
    finally:
        await client.close()
    assert len(requests) == 2
