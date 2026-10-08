from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from beaker.tracing import local_capture
from openai import APIStatusError, AsyncOpenAI
from verifiers.errors import ToolParseError

from automationbench_skills.clients import BoundedResponsesClient, CostTrackingChatCompletionsClient, inference_context
from automationbench_skills.runner import ModelSpec, get_env


def _reply(api: str, nested: str | None = None, *, response_id: str = "reply") -> dict[str, Any]:
    arguments = json.dumps({"tool_name": "google_sheets_get_many_rows", "arguments": nested})
    if api == "responses":
        return {
            "id": response_id,
            "object": "response",
            "created_at": 1,
            "model": "gpt-5.6-luna",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "execute_tool",
                    "arguments": arguments,
                }
            ],
            "usage": {
                "input_tokens": 11,
                "output_tokens": 7,
                "total_tokens": 18,
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    message: dict[str, Any] = {"role": "assistant", "content": "done"}
    if nested is not None:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "execute_tool", "arguments": arguments}}
            ],
        }
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5.6-luna",
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if nested is not None else "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18, "cost": 0.01},
    }


def _client(api: str, handler: Any, **kwargs: Any) -> Any:
    sdk = AsyncOpenAI(
        api_key="credential-never-logged",
        max_retries=9,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    cls = BoundedResponsesClient if api == "responses" else CostTrackingChatCompletionsClient
    return cls(sdk, **kwargs)


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
async def test_caps_reach_the_wire_and_parallel_calls_remain_optional(api: str) -> None:
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_reply(api, "{}"))

    model = ModelSpec(api=api, max_output_tokens=8192, parallel_tool_calls=False)
    client = _client(api, handler)
    try:
        await client.get_native_response([], model.name, model.sampling_args())
    finally:
        await client.close()
    (body,) = captured
    key = "max_output_tokens" if api == "responses" else "max_completion_tokens"
    assert body[key] == 8192
    assert body["parallel_tool_calls"] is False
    assert "parallel_tool_calls" not in ModelSpec(api=api).sampling_args()


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
async def test_corrupted_nested_json_is_retried_before_dispatch_and_usage_is_retained(api: str, capsys: Any) -> None:
    replies = [
        _reply(api, "{} trailing secret-data", response_id="bad"),
        _reply(api, '{"title":"ગુજરાતી 中文 José"}', response_id="good"),
    ]
    # Reproduce the observed batch: a valid first call, then a corrupt one.
    if api == "responses":
        replies[0]["output"].insert(0, _reply(api, "{}")["output"][0])
    else:
        calls = replies[0]["choices"][0]["message"]["tool_calls"]
        calls.insert(0, _reply(api, "{}")["choices"][0]["message"]["tool_calls"][0])
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=replies.pop(0), headers={"x-request-id": f"req-{len(requests)}"})

    state: dict[str, Any] = {}
    client = _client(api, handler)
    try:
        with inference_context("case-0020", "operations.policy"):
            response = await client.get_native_response(
                [], "gpt-5.6-luna", ModelSpec(api=api).sampling_args(), state=state
            )
            converted = await client.from_native_response(response)
    finally:
        await client.close()
    assert response.id == "good"
    assert len(converted.message.tool_calls) == 1
    assert len(requests) == 2
    assert requests[0] == requests[1]
    assert state["_usage"] == {"input_tokens": 11, "output_tokens": 7}
    assert state["_perf"]["model_calls"] == 2
    if api == "chat_completions":
        assert state["_perf"]["cost_usd"] == pytest.approx(0.02)
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["phase"] for record in records] == ["start", "response", "format_retry", "start", "response"]
    assert all(record["case_id"] == "case-0020" for record in records)
    assert len({record["rollout_id"] for record in records}) == 1
    assert records[1]["request_id"] == "req-1"
    assert records[1]["format_valid"] is False
    assert records[-1]["format_valid"] is True
    assert "secret-data" not in json.dumps(records)
    assert "credential-never-logged" not in json.dumps(records)


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
@pytest.mark.parametrize("nested", ["[]", '{"value":NaN}'])
async def test_persistent_invalid_json_gets_only_one_format_retry(api: str, nested: str) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_reply(api, nested))

    state: dict[str, Any] = {}
    client = _client(api, handler, max_attempts=5)
    try:
        with pytest.raises(ToolParseError):
            await client.get_native_response([], "gpt-5.6-luna", {}, state=state)
    finally:
        await client.close()
    assert len(requests) == 2
    assert state["_usage"] == {"input_tokens": 22, "output_tokens": 14}


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
@pytest.mark.parametrize(("status", "attempts"), [(503, 3), (429, 3), (400, 1), (401, 1)])
async def test_transport_retries_are_bounded_and_deterministic_errors_fail_fast(
    api: str, status: int, attempts: int
) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status, json={"error": {"message": "failed", "type": "api_error"}}, headers={"retry-after": "0"}
        )

    client = _client(api, handler, max_attempts=3)
    assert client.client.max_retries == 0
    try:
        with pytest.raises(APIStatusError):
            await client.get_native_response([], "gpt-5.6-luna", {})
    finally:
        await client.close()
    assert len(requests) == attempts


@pytest.mark.parametrize("backoff", [False, True])
async def test_deadline_covers_a_stalled_request_or_backoff(backoff: bool, capsys: Any) -> None:
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
        return httpx.Response(200, json=_reply("chat_completions"))

    client = _client("chat_completions", handler, request_timeout=0.03)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            await client.get_native_response([], "gpt-5.6-luna", {})
    finally:
        await client.close()
    assert time.monotonic() - started < 1
    assert len(requests) == 1
    assert backoff or cancelled.is_set()
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[0]["phase"] == "start"
    assert records[-1]["phase"] == "deadline"


async def test_hosted_rejected_and_accepted_responses_each_have_a_trace(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).parents[1] / ".beaker"))
    from beaker_integration import _TracedChatCompletionsClient

    replies = [_reply("chat_completions", "{}garbage"), _reply("chat_completions", '{"range":"A:Z"}')]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=replies.pop(0))

    client = _TracedChatCompletionsClient(
        AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    )
    try:
        with local_capture(tmp_path, case_id="case", candidate_id="candidate", strict_evidence=False) as capture:
            await client.get_native_response([], "gpt-5.6-luna", {"max_completion_tokens": 16384}, state={})
    finally:
        await client.close()
    assert capture.receipt is not None
    calls = capture.receipt.to_dict()["projection"]["model_calls"]
    assert len(calls) == 2
    assert sum(call["usage"]["output_tokens"] for call in calls) == 14


def test_extra_body_cannot_raise_the_configured_cap_and_native_presets_are_preserved() -> None:
    model = ModelSpec(extra_body='{"max_tokens":128000}')
    assert model.sampling_args()["max_completion_tokens"] == 16384
    assert "max_tokens" not in model.sampling_args()["extra_body"]
    lower = ModelSpec(api="responses", extra_body='{"max_completion_tokens":1024}')
    assert lower.sampling_args()["max_output_tokens"] == 1024
    anthropic = ModelSpec(name="claude-sonnet-5-5", api="anthropic", reasoning_effort="max")
    assert "max_completion_tokens" not in anthropic.sampling_args()


def test_search_limit_is_optional_and_has_a_distinct_cached_environment() -> None:
    baseline = get_env(skills=False)
    limited = get_env(skills=False, search_top_k=5)
    assert limited is get_env(skills=False, search_top_k=5)
    assert limited is not baseline


def test_cli_exposes_optional_experiments_and_records_defaults() -> None:
    import argparse

    from automationbench_skills.cli import _add_run_args

    parser = argparse.ArgumentParser()
    _add_run_args(parser)
    defaults = parser.parse_args([])
    assert defaults.max_output_tokens == 16384
    assert defaults.max_model_attempts == 3
    assert defaults.model_request_timeout == 300
    assert defaults.parallel_tool_calls is None
    assert defaults.search_top_k is None
    experiment = parser.parse_args(["--no-parallel-tool-calls", "--search-top-k", "5"])
    assert experiment.parallel_tool_calls is False
    assert experiment.search_top_k == 5


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_deadline_cannot_disable_the_guard(value: float) -> None:
    with pytest.raises(ValueError):
        ModelSpec(model_request_timeout=value)
