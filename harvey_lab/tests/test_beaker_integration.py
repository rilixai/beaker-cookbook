"""Offline regression checks for the Beaker adapter's provider boundary."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, Mock

import httpx
import litellm
import pytest
from beaker import RetryableCaseError, RolloutRuntime, SetupRuntime
from beaker.sdk.inference import InferenceTarget
from beaker.tracing import local_capture
from beaker.tracing.core import NoopTrace
from beaker.tracing.integrations.litellm import registered
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
@pytest.mark.parametrize("window", [None, 1_000_000, 16_384])
async def test_selected_model_request_omits_production_limits(
    integration: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, window: int | None
) -> None:
    if window is None:
        monkeypatch.delenv("HARVEY_BEAKER_CONTEXT_WINDOW_TOKENS", raising=False)
    else:
        monkeypatch.setenv("HARVEY_BEAKER_CONTEXT_WINDOW_TOKENS", str(window))
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
    assert client.context_window_tokens == (window or 128_000)
    assert client.max_tokens <= client.context_window_tokens
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        async with AsyncOpenAI(api_key="test-key", base_url=target.base_url, http_client=http_client) as api:
            # Keep Stirrup and LiteLLM's real request construction; replace only HTTP transport.
            client._kwargs["client"] = api
            async with local_capture(tmp_path / "trace") as capture:
                async with registered(capture.trace) as tracing:
                    await client.generate([UserMessage(content="Hello")], {})
                    assert await tracing.flush(timeout=5)
    assert len(requests) == 1
    assert requests[0]["model"] == target.model
    assert "max_tokens" not in requests[0]
    assert "max_completion_tokens" not in requests[0]
    assert "temperature" not in requests[0]
    assert "reasoning_effort" not in requests[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("window", ["0", "-1", "invalid", "1.5", "", "1M"])
async def test_context_window_rejected_during_setup_and_execution(
    integration: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, window: str
) -> None:
    monkeypatch.setenv("HARVEY_BEAKER_CONTEXT_WINDOW_TOKENS", window)
    target = InferenceTarget(base_url="https://gateway.example/v1", api_key="test-key", model="openai:test-model")
    monkeypatch.setattr(integration, "inference_target", lambda runtime: target)
    runtime = RolloutRuntime(case_files_dir=tmp_path, trace=NoopTrace(), model=target.model)
    with pytest.raises(ValueError, match="HARVEY_BEAKER_CONTEXT_WINDOW_TOKENS must be a positive integer"):
        integration.selected_model_factory(runtime)
    with pytest.raises(ValueError, match="HARVEY_BEAKER_CONTEXT_WINDOW_TOKENS must be a positive integer"):
        async with integration.TaskSetup().prepare_run(runtime=Mock(spec=SetupRuntime)):
            pytest.fail("Invalid context window must fail before setup yields")
