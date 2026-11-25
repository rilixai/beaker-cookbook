"""The request guard, the native request, and observable failures."""

from __future__ import annotations

import json
from typing import Any

import fake_api
import pytest

from decisionpivot import agent
from decisionpivot.data import Item, Question
from decisionpivot.decide import DecisionError, Request, RequestRejected, Session, guard, run_item


ORIGINAL = Question(
    type="choice",
    instructions="Which entry fails?",
    criteria={"e3": "Entry 3 fails.", "e1": "Entry 1 fails.", "none": "No entry fails."},
)


def edited(**criteria: str) -> Question:
    return ORIGINAL.model_copy(update={"criteria": criteria})


@pytest.mark.parametrize(
    "request_question",
    [
        edited(e3="Entry 3 fails.", e1="Entry 1 fails."),  # dropped
        edited(e1="Entry 1 fails.", e3="Entry 3 fails.", none="No entry fails."),  # reordered
        edited(e3="Entry 3 fails.", e1="Entry 1 fails.", none="No entry fails.", e2="Entry 2 fails."),  # added
        edited(e3="Entry 3 is fine.", e1="Entry 1 fails.", none="No entry fails."),  # option text edited
        ORIGINAL.model_copy(update={"instructions": "Which entry is fine?"}),  # question replaced
    ],
)
def test_guard_rejects(request_question: Question) -> None:
    with pytest.raises(RequestRejected):
        guard(ORIGINAL, request_question)


def test_guard_allows_added_text() -> None:
    guard(
        ORIGINAL,
        ORIGINAL.model_copy(
            update={
                "instructions": "Check each entry against the rules. Which entry fails?",
                "criteria": {"e3": "Entry 3 fails.", "e1": "Entry 1 fails. (row 1)", "none": "No entry fails."},
            }
        ),
    )


def item(state: Any = "ledger text") -> Item:
    return Item(id="ledger_reconcile:00001", state=state, questions={"q": ORIGINAL})


async def test_rejected_request_never_reaches_the_api() -> None:
    received: list[dict[str, Any]] = []

    async def reorder(it: Item, session: Session, instructions: str) -> dict[str, float]:
        swapped = edited(e1="Entry 1 fails.", e3="Entry 3 fails.", none="No entry fails.")
        bad = Request(state=it.state, question=swapped)
        return (await session.decide(bad)).probabilities

    outcome = await run_item(item(), solver=reorder, instructions="", client=fake_api.client(received))
    assert received == [] and outcome.probabilities is None
    assert outcome.error and outcome.error.startswith("RequestRejected")


async def test_seed_sends_state_and_question_unchanged() -> None:
    received: list[dict[str, Any]] = []
    state = {"request": "Find a lamp under 40 GBP.", "record": {"price": "39.50", "currency": "GBP"}}
    outcome = await run_item(item(state), solver=agent.answer, instructions="", client=fake_api.client(received))
    assert outcome.error is None
    assert len(received) == 1
    assert json.dumps(received[0]["state"]) == json.dumps(state)
    assert json.dumps(received[0]["questions"]) == json.dumps({"q": ORIGINAL.model_dump(mode="json")})
    assert received[0]["model"] == "jev-1.14.0"
    assert list(outcome.probabilities or {}) == ["e3", "e1", "none"]


async def test_instructions_go_in_front_of_the_question() -> None:
    received: list[dict[str, Any]] = []
    await run_item(item(), solver=agent.answer, instructions="Be careful.", client=fake_api.client(received))
    assert received[0]["questions"]["q"]["instructions"] == "Be careful.\n\nWhich entry fails?"


@pytest.mark.parametrize(
    ("answer", "status"),
    [
        (lambda q, s: {"type": "choice", "choice": "e3", "probabilities": {"e3": 1.0}}, "malformed_answer"),
        (lambda q, s: {"type": "choice", "probabilities": {"e3": 0.0, "e1": 0.0, "none": 0.0}}, "malformed_answer"),
    ],
)
async def test_malformed_answers_are_observable_errors(answer: Any, status: str) -> None:
    outcome = await run_item(item(), solver=agent.answer, instructions="", client=fake_api.client(answer=answer))
    assert outcome.probabilities is None and outcome.error and outcome.error.startswith(status)


async def test_server_errors_are_retryable_and_model_changes_are_not() -> None:
    failed = await run_item(item(), solver=agent.answer, instructions="", client=fake_api.client(status=503))
    assert failed.retryable and failed.error and failed.error.startswith("request_error")
    changed = await run_item(item(), solver=agent.answer, instructions="", client=fake_api.client(model="jev-9"))
    assert not changed.retryable and changed.error and "model_version_changed" in changed.error


async def test_answer_must_cover_every_option() -> None:
    async def partial(it: Item, session: Session, instructions: str) -> dict[str, float]:
        return {"e3": 1.0}

    outcome = await run_item(item(), solver=partial, instructions="")
    assert outcome.error and outcome.error.startswith("missing_option")


async def test_answer_must_come_from_a_decision_call() -> None:
    async def overrides(it: Item, session: Session, instructions: str) -> dict[str, float]:
        await session.decide(Request.of(it))
        return {"e3": 0.0, "e1": 1.0, "none": 0.0}  # e.g. a chat model's pick, replacing the decision

    async def no_call(it: Item, session: Session, instructions: str) -> dict[str, float]:
        return {"e3": 1.0, "e1": 0.0, "none": 0.0}

    async def picks_second(it: Item, session: Session, instructions: str) -> dict[str, float]:
        await session.decide(Request.of(it))
        second = await session.decide(Request.of(it))
        return {k: v * 2 for k, v in second.probabilities.items()}  # unnormalised is fine

    for solver in (overrides, no_call):
        outcome = await run_item(item(), solver=solver, instructions="", client=fake_api.client())
        assert outcome.probabilities is None and outcome.error and outcome.error.startswith("answer_not_from_decision")
    ok = await run_item(item(), solver=picks_second, instructions="", client=fake_api.client())
    assert ok.error is None and ok.probabilities is not None
    assert isinstance(DecisionError("x", "y"), RuntimeError)
