"""The Beaker integration: traced decision calls, model routing, failures stay observable."""

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

from alfworld_agent import cli, decide
from alfworld_agent.data import CACHE, SPLIT_FILES
from alfworld_agent.env import View


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
    spec = importlib.util.spec_from_file_location("alfworld_integration", ROOT / ".beaker/beaker_integration.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def load_case(integration: ModuleType) -> Case:
    setup = integration.Setup()
    async with setup.prepare_run(runtime=None):
        cases = [c async for c in setup.load_cases(fake_api.row(), runtime=None)]
    assert len(cases) == 1
    return cases[0]


def runtime(model: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(model=model, trace=RecordingTrace())


async def test_run_case_traces_each_decision_call_and_scores_a_win(
    integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = await load_case(integration)
    assert set(case.input) == {"id", "split", "game"} and case.expected == {"task_type": "pick_and_place_simple"}
    received: list[dict[str, Any]] = []
    monkeypatch.setattr(
        decide, "AsyncTypeSafeClient", lambda **kw: fake_api.client(received, choose=choose_walkthrough())
    )
    monkeypatch.chdir(ROOT)
    run = runtime()
    result = await integration.run_case(case_input=case.input, runtime=run)

    sent = json.dumps(received)
    for leak in ("tw-pddl", "trial_T", "pick_and_place_simple", "valid_seen", "walkthrough"):
        assert leak not in sent
    trace = run.trace
    assert len(trace.model_calls) == 4
    for call in trace.model_calls:
        assert call.attributes["operation"] == "systemone"
        assert call.attributes["provider"] == "typesafe"
        assert call.attributes["server_address"] == "api.typesafe.ai"
        assert call.attributes["model"] == decide.MODEL
        assert call.usages == [{"input_tokens": 500, "output_tokens": 2}]
    episode, *steps = trace.stages
    assert episode.attributes == {"name": "alfworld.episode", "inputs": {"id": case.id}}
    assert [s.attributes["name"] for s in steps] == ["alfworld.step"] * 4
    assert steps[0].attributes["inputs"]["step"] == 1 and steps[0].attributes["inputs"]["option_count"] > 2

    assert result.output["outcome"] == {"won": True, "steps": 4, "hit_step_limit": False, "error": None}
    assert len(result.output["last_steps"]) == 3 and len(result.context["transcript"]) == 4
    scored = await integration.score_case(case=case, result=result, case_files_dir=ROOT)
    assert scored.objective == 1.0 and scored.field_scores["success"] == 1.0
    assert [(c.name, c.verdict) for c in scored.checks] == [("success", "pass"), ("step_limit", "pass")]


def choose_walkthrough() -> fake_api.Chooser:
    return fake_api.scripted(fake_api.walkthrough())


async def test_a_model_choice_reroutes_decision_calls_through_the_gateway(
    integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = await load_case(integration)
    target = InferenceTarget(base_url="https://gateway.example", api_key="run-token", model="openrouter:decider-1")
    monkeypatch.setattr(integration, "inference_target", lambda runtime: target)
    monkeypatch.chdir(ROOT)
    run = runtime("openrouter:decider-1")
    with fake_api.patched_client(
        monkeypatch, lambda: fake_api.client(model="openrouter:decider-1", choose=choose_walkthrough())
    ) as built:
        result = await integration.run_case(case_input=case.input, runtime=run)
    assert built[0]["base_url"] == "https://gateway.example" and built[0]["api_key"] == "run-token"
    assert result.output["outcome"]["won"] and result.output["models"] == ["openrouter:decider-1"]
    call = run.trace.model_calls[0]
    assert call.attributes["provider"] == "openrouter" and call.attributes["model"] == "openrouter:decider-1"
    assert call.attributes["server_address"] == "gateway.example"


async def test_a_candidate_chat_call_is_traced_and_follows_the_run_model(
    integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = await load_case(integration)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        message = {"role": "assistant", "content": "look"}
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

    class Planner:
        def __init__(self, instructions: str) -> None:
            self.planned = False

        async def act(self, view: View, session: decide.Session) -> str:
            if not self.planned:
                self.planned = True
                await session.chat([{"role": "user", "content": view.goal}])
            return "look"

    target = InferenceTarget(base_url="https://gateway.example/v1", api_key="run-token", model="openai:gpt-test")
    monkeypatch.setattr(integration, "inference_target", lambda runtime: target)
    monkeypatch.setattr(integration, "AsyncOpenAI", openai_client)
    monkeypatch.setattr(integration.agent, "Agent", Planner)
    monkeypatch.chdir(ROOT)
    run = runtime("openai:gpt-test")
    result = await integration.run_case(case_input=case.input, runtime=run)
    assert result.output["chat_calls"] == 1 and result.output["outcome"]["hit_step_limit"]
    [chat] = run.trace.model_calls
    assert chat.attributes["operation"] == "chat" and chat.attributes["model"] == "openai:gpt-test"
    assert chat.usages == [{"input_tokens": 12, "output_tokens": 2}]
    assert str(requests[0].url) == "https://gateway.example/v1/chat/completions"
    scored = await integration.score_case(case=case, result=result, case_files_dir=ROOT)
    assert scored.objective == 0.0 and [(c.name, c.verdict) for c in scored.checks] == [
        ("success", "fail"),
        ("step_limit", "fail"),
    ]


async def test_request_errors_are_failed_cases_and_transient_errors_retry(
    integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = await load_case(integration)
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(decide, "AsyncTypeSafeClient", lambda **kw: fake_api.client(status=400))
    result = await integration.run_case(case_input=case.input, runtime=runtime())
    assert result.output["outcome"]["error"].startswith("request_error: TypeSafeBadRequestError")
    scored = await integration.score_case(case=case, result=result, case_files_dir=ROOT)
    assert scored.objective == 0.0 and scored.checks[0].verdict == "fail" and scored.checks[0].message

    monkeypatch.setattr(decide, "AsyncTypeSafeClient", lambda **kw: fake_api.client(status=429))
    with pytest.raises(RetryableCaseError):
        await integration.run_case(case_input=case.input, runtime=runtime())

    malformed = CaseResult(output={"outcome": {"won": True}}, output_kind="record")
    scored = await integration.score_case(case=case, result=malformed, case_files_dir=ROOT)
    assert scored.objective == 0.0 and scored.checks[0].message == "malformed_output"


async def test_cli_evaluate_writes_results_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / SPLIT_FILES["validation"]
    path.parent.mkdir(parents=True)
    path.write_text(fake_api.row().model_dump_json() + "\n")
    args = argparse.Namespace(
        data=tmp_path,
        split="validation",
        limit=None,
        prompts=ROOT / "prompts",
        concurrency=1,
        output=tmp_path / "out" / "validation.jsonl",
        cache=CACHE,
    )
    summary = await cli.evaluate(args, client=fake_api.client(choose=choose_walkthrough()))
    assert summary["episodes"] == 1 and summary["success_rate"] == 1.0 and summary["failed"] == 0
    assert summary["success_rate_by_task_type"] == {"pick_and_place_simple": {"episodes": 1, "success_rate": 1.0}}
    assert summary["mean_steps_on_success"] == 4 and summary["decision_calls_per_episode"] == 4
    assert summary["cost_per_episode_usd"] == pytest.approx(4 * 500 * decide.PRICE_PER_MILLION_INPUT_TOKENS / 1e6)
    [record] = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert record["id"] == fake_api.row().id and len(record["transcript"]) == 4
    assert json.loads(args.output.with_suffix(".summary.json").read_text()) == summary

    failing = await cli.evaluate(args, client=fake_api.client(status=400))
    assert failing["failed"] == 1 and failing["failed_by_status"] == {"request_error": 1}
