"""Offline regression checks for the Beaker adapter's provider boundary."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock

import httpx
import litellm
import pytest
from beaker import RetryableCaseError, RolloutRuntime
from beaker.sdk.inference import InferenceTarget
from beaker.tracing.core import NoopTrace
from openai import AsyncOpenAI
from stirrup.clients import litellm_client
from stirrup.clients.litellm_client import LiteLLMClient
from stirrup.core.models import UserMessage
from tenacity import RetryError, retry, stop_after_attempt, wait_none

from harvey_lab.agent.agent import HarveyLabAgent


@pytest.fixture
def integration(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / ".beaker"))
    return importlib.import_module("beaker_integration")


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [litellm.RateLimitError, litellm.Timeout, litellm.APIConnectionError])
async def test_exhausted_stirrup_retries_remain_retryable(
    integration: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error_type: type[Exception]
) -> None:
    error = error_type(message="transient provider failure", model="test-model", llm_provider="openai")
    completion = AsyncMock(side_effect=error)
    monkeypatch.setattr(litellm_client, "acompletion", completion)
    monkeypatch.setattr(LiteLLMClient.generate.retry, "wait", wait_none())
    client = LiteLLMClient(model="openai/test-model", max_tokens=100, context_window_tokens=1000)

    async def forward(self: HarveyLabAgent, *, record: object) -> None:
        await client.generate([UserMessage(content="Evaluate this task")], {})

    monkeypatch.setattr(HarveyLabAgent, "forward", forward)
    runtime = RolloutRuntime(case_files_dir=tmp_path, trace=NoopTrace())
    task = integration.TaskInput(task_id="contracts/example", title="Example", instructions="Review", deliverables={})
    with pytest.raises(RetryableCaseError) as caught:
        await integration.run_case(case_input=task.model_dump(mode="json"), runtime=runtime)
    assert completion.await_count == 3
    assert isinstance(caught.value.__cause__, RetryError)
    assert caught.value.__cause__.last_attempt.exception() is error


@pytest.mark.asyncio
async def test_non_transient_retry_error_is_not_reclassified(
    integration: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    @retry(stop=stop_after_attempt(1), wait=wait_none())
    async def forward(self: HarveyLabAgent, *, record: object) -> None:
        raise ValueError("invalid task")

    monkeypatch.setattr(HarveyLabAgent, "forward", forward)
    runtime = RolloutRuntime(case_files_dir=tmp_path, trace=NoopTrace())
    task = integration.TaskInput(task_id="contracts/example", title="Example", instructions="Review", deliverables={})
    with pytest.raises(RetryError) as caught:
        await integration.run_case(case_input=task.model_dump(mode="json"), runtime=runtime)
    assert isinstance(caught.value.last_attempt.exception(), ValueError)


@pytest.mark.asyncio
async def test_selected_model_request_omits_production_limits(
    integration: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = InferenceTarget(base_url="https://gateway.example/v1", api_key="test-key", model="openai:test-model")
    monkeypatch.setattr(integration, "inference_target", lambda runtime: target)
    requests: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "test-completion",
                "object": "chat.completion",
                "created": 0,
                "model": target.model,
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    runtime = RolloutRuntime(case_files_dir=tmp_path, trace=NoopTrace(), model=target.model)
    factory = integration.selected_model_factory(runtime)
    client = factory("production-model", 0.6, 384_000, 1_000_000, 60.0, "xhigh")
    assert isinstance(client, LiteLLMClient)
    assert client.context_window_tokens == 32_768
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        async with AsyncOpenAI(api_key="test-key", base_url=target.base_url, http_client=http_client) as api:
            # Keep Stirrup and LiteLLM's real request construction; replace only HTTP transport.
            client._kwargs["client"] = api
            await client.generate([UserMessage(content="Hello")], {})
    assert len(requests) == 1
    assert requests[0]["model"] == target.model
    assert "max_tokens" not in requests[0]
    assert "max_completion_tokens" not in requests[0]
    assert "temperature" not in requests[0]
    assert "reasoning_effort" not in requests[0]
