"""The Beaker integration: one agent call per item, never siblings or labels; traced calls; observable failures."""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import fake_api
import pytest
from beaker import Case, CaseResult, RetryableCaseError

from decisionpivot import agent, decide
from decisionpivot.data import Item
from decisionpivot.decide import Session


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
    spec = importlib.util.spec_from_file_location("decisionpivot_integration", ROOT / ".beaker/beaker_integration.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def load_case(integration: ModuleType, golds: tuple[str, ...] = ("yes", "no", "yes")) -> Case:
    cases = [c async for c in integration.Setup().load_cases(fake_api.row(golds=golds), runtime=None)]
    assert len(cases) == 1
    return cases[0]


def runtime(trace: RecordingTrace, model: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(model=model, trace=trace)


async def test_run_case_calls_the_agent_once_per_item_and_never_passes_siblings(
    integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = await load_case(integration)
    seen: list[Item] = []

    async def stub(item: Item, session: Session, instructions: str) -> dict[str, float]:
        seen.append(item)
        assert session.item == item  # the session holds this item only
        return {"yes": 0.9, "no": 0.1}

    monkeypatch.setattr(agent, "answer", stub)
    monkeypatch.chdir(ROOT)
    result = await integration.run_case(case_input=case.input, runtime=runtime(RecordingTrace()))
    assert [i.id for i in seen] == [i["id"] for i in case.input["items"]]
    assert all(set(i.model_dump()) == {"id", "state", "questions"} for i in seen)
    assert len({i.state for i in seen}) == len(seen)
    assert [o["id"] for o in result.output["items"]] == [i.id for i in seen]


async def test_case_input_has_no_labels(integration: ModuleType) -> None:
    case = await load_case(integration)
    assert set(case.input) == {"group_id", "family", "items"}
    assert case.input["group_id"] == "sched_conflict:g0001"
    assert set(case.expected) == {"gold", "level", "layer"}
    assert case.metadata == {"family": "sched_conflict", "level": 3, "layer": "D"}


async def test_traces_one_decision_call_per_item_and_group_stages(
    integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = await load_case(integration)
    received: list[dict[str, Any]] = []
    monkeypatch.setattr(decide, "AsyncTypeSafeClient", lambda **kwargs: fake_api.client(received))
    monkeypatch.chdir(ROOT)
    trace = RecordingTrace()
    result = await integration.run_case(case_input=case.input, runtime=runtime(trace))
    assert len(received) == 3 and {r["model"] for r in received} == {"jev-1.14.0"}
    assert [c.attributes["operation"] for c in trace.model_calls] == ["systemone"] * 3
    call = trace.model_calls[0]
    assert call.attributes["provider"] == "typesafe" and call.attributes["server_address"] == "api.typesafe.ai"
    assert set(call.attributes["input_messages"]) == {"state", "questions"}
    assert call.usages == [{"input_tokens": 1000, "output_tokens": 3}] and call.outputs
    group, *items = trace.stages
    assert group.attributes == {
        "name": "decisionpivot.group",
        "inputs": {"group_id": "sched_conflict:g0001", "family": "sched_conflict", "size": 3},
    }
    assert [s.attributes["name"] for s in items] == ["decisionpivot.item"] * 3
    for payload in received:  # no labels, level or sibling text in any request
        text = str(payload)
        assert "level" not in text and "gold" not in text
    states = [r["state"] for r in received]
    assert len(set(states)) == 3
    scored = await integration.score_case(case=case, result=result, case_files_dir=ROOT)
    # the fake leans to the first option ("yes"): items 1 and 3 right, item 2 wrong
    assert scored.objective == 0.0 and scored.field_scores["item_accuracy"] == pytest.approx(2 / 3)
    assert [c.verdict for c in scored.checks] == ["pass", "fail", "pass"]
    assert [(c.expected, c.predicted) for c in scored.checks] == [("yes", "yes"), ("no", "yes"), ("yes", "yes")]


async def test_decision_calls_stay_native_when_a_chat_model_is_selected(
    integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = await load_case(integration)
    clients: list[dict[str, Any]] = []

    def make_client(**kwargs: Any) -> Any:
        clients.append(kwargs)
        return fake_api.client()

    monkeypatch.setattr(decide, "AsyncTypeSafeClient", make_client)
    monkeypatch.chdir(ROOT)
    trace = RecordingTrace()
    result = await integration.run_case(case_input=case.input, runtime=runtime(trace, "openai:gpt-6-luna"))
    assert all(o["error"] is None for o in result.output["items"])
    assert {c["base_url"] for c in clients} == {None}  # the SDK default: api.typesafe.ai, proxied by Beaker
    assert {c.attributes["model"] for c in trace.model_calls} == {"jev-1.14.0"}
    assert {c.attributes["server_address"] for c in trace.model_calls} == {"api.typesafe.ai"}


async def test_swapped_replies_naming_another_model_are_accepted(
    integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In a decision-model run the gateway answers the Jev call with the target's reply and model name."""
    case = await load_case(integration)
    monkeypatch.setattr(decide, "AsyncTypeSafeClient", lambda **kwargs: fake_api.client(model="gpt-6-luna"))
    monkeypatch.chdir(ROOT)
    result = await integration.run_case(case_input=case.input, runtime=runtime(RecordingTrace()))
    assert all(o["error"] is None for o in result.output["items"])
    assert {m for o in result.output["items"] for m in o["models"]} == {"gpt-6-luna"}


async def test_transient_errors_retry_and_other_failures_score_as_wrong(
    integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = await load_case(integration)
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(decide, "AsyncTypeSafeClient", lambda **kwargs: fake_api.client(status=503))
    with pytest.raises(RetryableCaseError):
        await integration.run_case(case_input=case.input, runtime=runtime(RecordingTrace()))

    async def broken(item: Item, session: Session, instructions: str) -> dict[str, float]:
        raise KeyError("candidate bug")

    monkeypatch.setattr(agent, "answer", broken)
    result = await integration.run_case(case_input=case.input, runtime=runtime(RecordingTrace()))
    assert all(o["error"].startswith("KeyError") for o in result.output["items"])
    scored = await integration.score_case(case=case, result=result, case_files_dir=ROOT)
    assert scored.objective == 0.0 and all(c.message for c in scored.checks)
    bad = await integration.score_case(case=case, result=CaseResult(output="nonsense"), case_files_dir=ROOT)
    assert bad.objective == 0.0
