"""Offline tests: no paid model calls. Pinned public upstream data is downloaded once and cached."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
import tau2.evaluator.evaluator_nl_assertions as upstream_judge
import tau2.user.user_simulator as upstream_user
from beaker import Case, CaseResult, RetryableCaseError
from beaker.tracing import current_trace
from litellm import ServiceUnavailableError
from pydantic import ValidationError
from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import (
    CommunicateCheck,
    DBCheck,
    NLAssertionCheck,
    RewardInfo,
    SimulationRun,
    TerminationReason,
)
from tau2.data_model.tasks import RewardType

from tau2_bench import benchmark, data, scoring
from tau2_bench.user_sim import USER_LLM


ROOT = Path(__file__).resolve().parents[1]
GATEWAY = "https://gateway.test/v1"


@pytest.fixture(scope="module")
def integration() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tau2_integration", ROOT / ".beaker/beaker_integration.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- data and splits ---------------------------------------------------------------------------


def synthetic(train: int, test: int) -> tuple[list[str], dict[str, list[str]]]:
    ids = [str(i) for i in range(train + test)]
    return ids, {"train": ids[:train], "test": ids[train:]}


def test_splits_are_deterministic_disjoint_and_hold_out_upstream_test() -> None:
    airline, retail = synthetic(30, 20), synthetic(74, 40)
    task_ids = {"airline": airline[0], "retail": retail[0]}
    official = {"airline": airline[1], "retail": retail[1]}
    splits = data.make_splits(task_ids, official)
    assert splits == data.make_splits(task_ids, official)
    assert {name: len(rows) for name, rows in splits.items()} == {"train": 77, "validation": 27, "test": 60}
    ids = {name: {row.id for row in rows} for name, rows in splits.items()}
    assert not (ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"])
    assert ids["test"] == {f"airline-{i}" for i in official["airline"]["test"]} | {
        f"retail-{i}" for i in official["retail"]["test"]
    }
    assert data.SPLIT_FILES["validation"] == "optimization/test.jsonl"


def test_splits_reject_upstream_files_that_do_not_partition_tasks() -> None:
    ids, official = synthetic(3, 2)
    with pytest.raises(ValueError, match="partition"):
        data.make_splits(
            {"airline": ids, "retail": ids}, {"airline": official, "retail": {"train": ids[:3], "test": ids[2:]}}
        )


def test_prepare_counts_match_manifest(tmp_path: Path) -> None:
    counts = data.prepare(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["counts"] == counts
    assert manifest["upstream_commit"] == data.UPSTREAM_COMMIT
    assert counts == {
        "train": {"airline": 22, "retail": 55, "total": 77},
        "validation": {"airline": 8, "retail": 19, "total": 27},
        "test": {"airline": 20, "retail": 40, "total": 60},
    }
    for split, path in data.SPLIT_FILES.items():
        assert len(data.load_rows(tmp_path / path)) == counts[split]["total"]


def test_corrupt_download_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAU2_DATA_DIR", str(tmp_path))

    class Corrupt:
        def __enter__(self) -> Corrupt:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return b"corrupt"

    monkeypatch.setattr(data, "urlopen", lambda *_args, **_kwargs: Corrupt())
    with pytest.raises(ValueError, match="Checksum mismatch"):
        data.ensure_data()
    assert not any(tmp_path.rglob("*.json"))


async def test_rows_validate_and_unknown_tasks_fail(integration: ModuleType) -> None:
    with pytest.raises(ValidationError):
        data.Row.model_validate({"id": "x", "input": {"domain": "telecom", "task_id": "0"}})
    with pytest.raises(ValidationError):
        data.Row.model_validate({"id": "x", "input": {"domain": "airline", "task_id": "0"}, "expected": {"a": 1}})
    setup = integration.Setup()
    cases = [case async for case in setup.load_cases(data.row_for("airline", "0"), runtime=None)]
    assert cases[0].input == {"domain": "airline", "task_id": "0"}
    with pytest.raises(KeyError, match="unknown tau2 airline"):
        async for _ in setup.load_cases(data.row_for("airline", "9999"), runtime=None):
            pass


# --- run_case: the agent follows runtime.model, the simulator never does -----------------------


def completion(model: str, content: str) -> dict[str, Any]:
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    }


@pytest.fixture
def stubbed(
    integration: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[dict[str, list[Any]]]:
    """Stub both model clients and run inside a candidate directory with its own prompt."""
    seen: dict[str, list[Any]] = {"agent": [], "user": [], "status": [200]}
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts/system.md").write_text("Candidate instructions: refuse politely.")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.setenv("BEAKER_INFERENCE_BASE_URL", GATEWAY)
    monkeypatch.setenv("BEAKER_INFERENCE_API_KEY", "run-token")

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen["agent"].append({"url": str(request.url), "auth": request.headers["authorization"], **body})
        status = seen["status"][0]
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "stub", "type": "stub"}})
        return httpx.Response(200, json=completion(body["model"], "Sorry, I cannot help with that."))

    real_openai = openai.OpenAI

    def client(**kwargs: Any) -> openai.OpenAI:
        return real_openai(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))

    monkeypatch.setattr(integration.openai, "OpenAI", client)
    monkeypatch.setattr(benchmark, "OpenAI", client)

    def user_generate(model: str, messages: list[Any], **kwargs: Any) -> AssistantMessage:
        seen["user"].append({"model": model, **kwargs})
        # Stop once the agent has answered (its turns reach the simulator with roles flipped).
        answered = "Sorry" in (messages[-1].content or "")
        content = "###STOP###" if answered else "Hi, please cancel my reservation."
        return AssistantMessage(role="assistant", content=content, usage={"prompt_tokens": 1, "completion_tokens": 1})

    monkeypatch.setattr(upstream_user, "generate", user_generate)
    yield seen


async def run(integration: ModuleType, model: str | None, task_id: str = "0") -> CaseResult:
    runtime = SimpleNamespace(model=model, trace=current_trace())
    result: CaseResult = await integration.run_case(
        case_input={"domain": "airline", "task_id": task_id}, runtime=runtime
    )
    return result


async def test_agent_follows_runtime_model_and_simulator_stays_pinned(
    integration: ModuleType, stubbed: dict[str, list[Any]]
) -> None:
    result = await run(integration, "openai:gpt-test")
    agent_calls, user_calls = stubbed["agent"], stubbed["user"]
    assert agent_calls and all(call["url"] == f"{GATEWAY}/chat/completions" for call in agent_calls)
    assert {call["model"] for call in agent_calls} == {"openai:gpt-test"}
    assert {call["auth"] for call in agent_calls} == {"Bearer run-token"}
    assert "reasoning_effort" not in agent_calls[0]
    assert "Candidate instructions: refuse politely." in agent_calls[0]["messages"][0]["content"]
    assert {call["model"] for call in user_calls} == {USER_LLM}
    assert result.output["termination_reason"] == "user_stop"  # type: ignore[index, call-overload]

    stubbed["agent"].clear()
    stubbed["user"].clear()
    await run(integration, None)
    assert {call["model"] for call in stubbed["agent"]} == {benchmark.DEFAULT_AGENT_MODEL}
    assert stubbed["agent"][0]["reasoning_effort"] == benchmark.DEFAULT_AGENT_REASONING
    assert {call["model"] for call in stubbed["user"]} == {USER_LLM}


async def test_trajectory_is_scored_by_upstream_evaluator(
    integration: ModuleType, stubbed: dict[str, list[Any]]
) -> None:
    # Airline task 0 expects a refusal and no database change: the stub agent passes it.
    result = await run(integration, "openai:gpt-test", "0")
    case = Case(id="airline-0", input={"domain": "airline", "task_id": "0"}, expected={})
    passed = await integration.score_case(case=case, result=result, case_files_dir=Path())
    assert passed.objective == 1.0
    assert dict(passed.field_scores) == {"reward": 1.0, "db": 1.0, "communicate": 1.0}
    # Airline task 3 requires communicating a value; refusing fails that check.
    result = await run(integration, "openai:gpt-test", "3")
    case = Case(id="airline-3", input={"domain": "airline", "task_id": "3"}, expected={})
    failed = await integration.score_case(case=case, result=result, case_files_dir=Path())
    assert failed.objective == 0.0
    communicate = [check for check in failed.checks if check.group == "COMMUNICATE"]
    assert communicate and communicate[0].verdict == "fail" and communicate[0].expected


async def test_provider_outage_retries_and_agent_errors_fail_visibly(
    integration: ModuleType, stubbed: dict[str, list[Any]]
) -> None:
    stubbed["status"][0] = 503
    with pytest.raises(RetryableCaseError):
        await run(integration, "openai:gpt-test")
    stubbed["status"][0] = 400
    result = await run(integration, "openai:gpt-test")
    assert "BadRequestError" in str(result.context["error"])
    case = Case(id="airline-0", input={"domain": "airline", "task_id": "0"}, expected={})
    scored = await integration.score_case(case=case, result=result, case_files_dir=Path())
    assert scored.objective == 0.0
    assert dict(scored.field_scores) == {"reward": 0.0, "db": 0.0, "communicate": 0.0}
    assert scored.checks[0].verdict == "fail" and "BadRequestError" in str(scored.checks[0].message)


async def test_simulator_outage_retries(
    integration: ModuleType, stubbed: dict[str, list[Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(**kwargs: Any) -> AssistantMessage:
        raise ServiceUnavailableError(message="simulator unavailable", llm_provider="openai", model=kwargs["model"])

    monkeypatch.setattr(upstream_user, "generate", unavailable)
    with pytest.raises(RetryableCaseError, match="simulator unavailable"):
        await run(integration, "openai:gpt-test")
    assert not stubbed["agent"]


# --- scoring map -------------------------------------------------------------------------------


def simulation(reason: TerminationReason, task_id: str = "1") -> SimulationRun:
    return SimulationRun(
        id="s",
        task_id=task_id,
        start_time="t",
        end_time="t",
        duration=0.0,
        termination_reason=reason,
        messages=[AssistantMessage(role="assistant", content="Done.")],
    )


@pytest.mark.parametrize("verdicts", ["missing", "duplicate", "unrelated", "complete"])
async def test_retail_requires_one_verdict_per_assertion(
    integration: ModuleType, monkeypatch: pytest.MonkeyPatch, verdicts: str
) -> None:
    # This pinned retail task requires an assertion and no database changes.
    task_id = "24"
    (assertion,) = benchmark.task("retail", task_id).evaluation_criteria.nl_assertions
    matching = {"expectedOutcome": assertion, "metExpectation": True, "reasoning": "Stub verdict."}
    judge_results = {
        "missing": [],
        "duplicate": [matching, matching],
        "unrelated": [{**matching, "expectedOutcome": "Unrelated assertion."}],
        "complete": [matching],
    }[verdicts]
    reply = AssistantMessage(role="assistant", content=json.dumps({"results": judge_results}))
    monkeypatch.setattr(upstream_judge, "generate", lambda **_: reply)
    case = Case(id=f"retail-{task_id}", input={"domain": "retail", "task_id": task_id}, expected={})
    recorded = scoring.to_context(simulation(TerminationReason.USER_STOP, task_id))
    result = CaseResult(output=None, output_kind="none", context={"simulation": recorded})
    if verdicts == "complete":
        scored = await integration.score_case(case=case, result=result, case_files_dir=Path())
        assert scored.objective == 1.0
    else:
        with pytest.raises(ValueError, match="one verdict per expected assertion"):
            await integration.score_case(case=case, result=result, case_files_dir=Path())


def test_retail_reward_maps_db_and_nl_assertions() -> None:
    basis = [RewardType.DB, RewardType.NL_ASSERTION]
    reward = RewardInfo(
        reward=0.0,
        db_check=DBCheck(db_match=True, db_reward=1.0),
        nl_assertions=[NLAssertionCheck(nl_assertion="Agent confirms first.", met=False, justification="It did not.")],
        reward_basis=basis,
        reward_breakdown={RewardType.DB: 1.0, RewardType.NL_ASSERTION: 0.0},
    )
    scored = scoring.score(reward, simulation(TerminationReason.USER_STOP), basis)
    assert scored.objective == 0.0
    assert dict(scored.field_scores) == {"reward": 0.0, "db": 1.0, "nl_assertion": 0.0}
    by_group = {check.group: check for check in scored.checks}
    assert by_group["DB"].verdict == "pass"
    assert by_group["NL_ASSERTION"].verdict == "fail"
    assert by_group["NL_ASSERTION"].message == "It did not."


def test_premature_end_is_a_failed_case_with_zeroed_components() -> None:
    basis = [RewardType.DB, RewardType.COMMUNICATE]
    reward = RewardInfo(reward=0.0, reward_basis=None, info={"note": "Simulation terminated prematurely."})
    scored = scoring.score(reward, simulation(TerminationReason.MAX_STEPS), basis)
    assert scored.objective == 0.0
    assert dict(scored.field_scores) == {"reward": 0.0, "db": 0.0, "communicate": 0.0}
    assert scored.checks[0].verdict == "fail" and scored.checks[0].predicted == "max_steps"


def test_communicate_checks_carry_expected_strings() -> None:
    basis = [RewardType.DB, RewardType.COMMUNICATE]
    reward = RewardInfo(
        reward=1.0,
        db_check=DBCheck(db_match=True, db_reward=1.0),
        communicate_checks=[CommunicateCheck(info="327", met=True, justification="")],
        reward_basis=basis,
        reward_breakdown={RewardType.DB: 1.0, RewardType.COMMUNICATE: 1.0},
    )
    scored = scoring.score(reward, simulation(TerminationReason.AGENT_STOP), basis)
    assert scored.objective == 1.0
    (check,) = [c for c in scored.checks if c.group == "COMMUNICATE"]
    assert check.expected == "327" and check.verdict == "pass"
