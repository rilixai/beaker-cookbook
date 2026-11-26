"""The Beaker integration: traced decision calls, no labels in requests, failures stay observable."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import fake_api
import httpx
import pytest
from beaker import Case, CaseResult, InferenceTarget, RetryableCaseError
from openai import AsyncOpenAI

from mmdm import cli, decide
from mmdm.data import SPLIT_FILES


ROOT = Path(__file__).resolve().parents[1]


class Operation:
    def __init__(self, **attributes: Any) -> None:
        self.attributes = attributes
        self.outputs: list[Any] = []
        self.usages: list[dict[str, Any]] = []

    def output(self, value: Any) -> None:
        self.outputs.append(value)

    def usage(self, **kwargs: Any) -> None:
        self.usages.append(kwargs)


class RecordingTrace:
    def __init__(self) -> None:
        self.model_calls: list[Operation] = []
        self.stages: list[Operation] = []

    @contextmanager
    def model_call(self, **attributes: Any) -> Iterator[Operation]:
        self.model_calls.append(Operation(**attributes))
        yield self.model_calls[-1]

    @contextmanager
    def stage(self, name: str, *, inputs: Any = None, attributes: Any = None) -> Iterator[Operation]:
        self.stages.append(Operation(name=name, inputs=inputs))
        yield self.stages[-1]


@pytest.fixture
def integration() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mmdm_integration", ROOT / ".beaker/beaker_integration.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def load_case(integration: ModuleType) -> Case:
    cases = [c async for c in integration.Setup().load_cases(fake_api.row(), runtime=None)]
    assert len(cases) == 1
    return cases[0]


async def test_run_case_traces_one_decision_call_and_never_sends_labels(
    integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = await load_case(integration)
    assert set(case.input) == {"id", "state", "question", "options", "task", "score_values"}
    received: list[dict[str, Any]] = []
    monkeypatch.setattr(decide, "AsyncTypeSafeClient", lambda **kwargs: fake_api.client(received))
    monkeypatch.chdir(ROOT)
    trace = RecordingTrace()
    result = await integration.run_case(case_input=case.input, runtime=SimpleNamespace(model=None, trace=trace))

    assert len(received) == 1 and received[0]["model"] == decide.MODEL
    sent = json.dumps(received)
    for leak in ("synthetic_category_xyz", "synthetic_subcategory_xyz", "synthetic_source_xyz", "correct_option_ids"):
        assert leak not in sent
    assert received[0]["questions"]["decision"]["instructions"].endswith("Which step should the agent retry?")
    [call] = trace.model_calls
    assert call.attributes["operation"] == "systemone"
    assert call.attributes["provider"] == "typesafe"
    assert call.attributes["server_address"] == "api.typesafe.ai"
    assert call.attributes["model"] == decide.MODEL
    assert call.usages == [{"input_tokens": 1000, "output_tokens": 3}]
    [stage] = trace.stages
    assert stage.attributes == {
        "name": "mmdm.question",
        "inputs": {"id": "mmdm-1", "task": "choice", "option_count": 3},
    }

    assert result.output["error"] is None and result.output["decision_calls"] == 1
    scored = await integration.score_case(case=case, result=result, case_files_dir=ROOT)
    assert scored.objective == 1.0 and scored.field_scores["hard_accuracy"] == 1.0
    assert scored.checks[0].expected == ["Retry migration"] and scored.checks[0].verdict == "pass"


async def test_a_model_choice_reroutes_decision_calls_through_the_gateway(
    integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = await load_case(integration)
    target = InferenceTarget(base_url="https://gateway.example", api_key="run-token", model="openrouter:decider-1")
    monkeypatch.setattr(integration, "inference_target", lambda runtime: target)
    built: list[dict[str, Any]] = []

    def client(**kwargs: Any) -> Any:
        built.append(kwargs)
        return fake_api.client()

    monkeypatch.setattr(decide, "AsyncTypeSafeClient", client)
    monkeypatch.chdir(ROOT)
    trace = RecordingTrace()
    await integration.run_case(
        case_input=case.input, runtime=SimpleNamespace(model="openrouter:decider-1", trace=trace)
    )
    assert built[0]["base_url"] == "https://gateway.example" and built[0]["api_key"] == "run-token"
    assert trace.model_calls[0].attributes["provider"] == "openrouter"
    assert trace.model_calls[0].attributes["model"] == "openrouter:decider-1"
    assert trace.model_calls[0].attributes["server_address"] == "gateway.example"


async def test_a_candidate_chat_call_is_traced_and_follows_the_run_model(
    integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = await load_case(integration)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        message = {"role": "assistant", "content": "Retry migration"}
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "created": 0,
                "model": "openai:gpt-test",
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
            },
        )

    def openai_client(**kwargs: Any) -> AsyncOpenAI:
        return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))

    async def candidate(question: Any, session: decide.Session, instructions: str) -> dict[str, float]:
        hint = await session.chat([{"role": "user", "content": question.state}])
        decision = await session.decide(question.model_copy(update={"question": hint + "\n" + question.question}))
        return decision.probabilities

    target = InferenceTarget(base_url="https://gateway.example/v1", api_key="run-token", model="openai:gpt-test")
    monkeypatch.setattr(integration, "inference_target", lambda runtime: target)
    monkeypatch.setattr(integration, "AsyncOpenAI", openai_client)
    monkeypatch.setattr(integration.agent, "answer", candidate)
    monkeypatch.setattr(decide, "AsyncTypeSafeClient", lambda **kwargs: fake_api.client())
    monkeypatch.chdir(ROOT)
    trace = RecordingTrace()
    result = await integration.run_case(
        case_input=case.input, runtime=SimpleNamespace(model="openai:gpt-test", trace=trace)
    )
    assert result.output["error"] is None and result.output["chat_calls"] == 1
    chat = trace.model_calls[0]
    assert chat.attributes["operation"] == "chat" and chat.attributes["model"] == "openai:gpt-test"
    assert chat.usages == [{"input_tokens": 12, "output_tokens": 2}]
    assert str(requests[0].url) == "https://gateway.example/v1/chat/completions"


async def test_request_errors_are_failed_cases_and_transient_errors_retry(
    integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = await load_case(integration)
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(decide, "AsyncTypeSafeClient", lambda **kwargs: fake_api.client(status=400))
    result = await integration.run_case(
        case_input=case.input, runtime=SimpleNamespace(model=None, trace=RecordingTrace())
    )
    assert result.output["probabilities"] is None
    # The server's status and message reach the case record, not just the exception class.
    assert result.output["error"].startswith("request_error: TypeSafeBadRequestError: ")
    assert "400" in result.output["error"]
    scored = await integration.score_case(case=case, result=result, case_files_dir=ROOT)
    assert scored.objective == 0.0 and scored.checks[0].verdict == "fail" and scored.checks[0].message

    monkeypatch.setattr(decide, "AsyncTypeSafeClient", lambda **kwargs: fake_api.client(status=429))
    with pytest.raises(RetryableCaseError):
        await integration.run_case(case_input=case.input, runtime=SimpleNamespace(model=None, trace=RecordingTrace()))

    malformed = CaseResult(output={"probabilities": {"o1": 1.0}, "error": None}, output_kind="record")
    scored = await integration.score_case(case=case, result=malformed, case_files_dir=ROOT)
    assert scored.objective == 0.0 and scored.checks[0].message == "missing_option"


async def test_cli_evaluate_writes_predictions_summary_and_matches_upstream(tmp_path: Path) -> None:
    rows = [fake_api.row(i) for i in range(1, 5)]
    path = tmp_path / SPLIT_FILES["validation"]
    path.parent.mkdir(parents=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in rows))
    args = argparse.Namespace(
        data=tmp_path,
        split="validation",
        limit=3,
        prompts=ROOT / "prompts",
        concurrency=2,
        output=tmp_path / "out" / "validation.jsonl",
        model=None,
        base_url=None,
        api_key_env=None,
        expected_model=None,
    )
    summary = await cli.evaluate(args, client=fake_api.client())
    assert summary["cases"] == 3 and summary["hard_accuracy"] == 1.0
    assert summary["coverage"] == {"native": 3, "grouped": 0, "failed": 0, "failed_by_status": {}}
    assert summary["upstream_check"]["matches"]
    assert summary["cost_per_1000_questions_usd"] == pytest.approx(
        1000 * 1000 * decide.PRICE_PER_MILLION_INPUT_TOKENS / 1e6
    )
    records = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert [r["id"] for r in records] == ["mmdm-1", "mmdm-2", "mmdm-3"]
    predictions = args.output.with_suffix(".upstream-predictions.jsonl").read_text().splitlines()
    assert json.loads(predictions[0])["id"] == 1

    failing = await cli.evaluate(args, client=fake_api.client(status=400))
    assert failing["coverage"]["failed"] == 3 and failing["hard_accuracy"] == 0.0
    assert failing["upstream_check"]["matches"]
