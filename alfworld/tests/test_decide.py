"""The request guard and the decision call."""

from __future__ import annotations

import fake_api
import pytest

from alfworld_agent.decide import OPTION_CAP, DecisionError, RequestRejected, Session, guard


ADMISSIBLE = ("go to bed 1", "go to desk 1", "inventory", "look")


def test_the_guard_rejects_an_option_that_is_not_admissible() -> None:
    with pytest.raises(RequestRejected, match="not admissible"):
        guard({"go to bed 1": "go to bed 1", "fly to the moon": "fly to the moon"}, ADMISSIBLE)


def test_the_guard_allows_leaving_options_out_and_custom_descriptions() -> None:
    guard({"go to bed 1": "go to bed 1 (not visited yet)", "go to desk 1": "go to desk 1"}, ADMISSIBLE)


@pytest.mark.parametrize(
    "criteria",
    [
        {"look": "look"},
        {"look": "look", "inventory": " "},
        {f"go to drawer {i}": "x" for i in range(OPTION_CAP + 1)},
    ],
)
def test_the_guard_rejects_malformed_requests(criteria: dict[str, str]) -> None:
    with pytest.raises(RequestRejected):
        guard(criteria, (*ADMISSIBLE, *criteria))


async def test_a_rejected_request_never_reaches_the_api() -> None:
    received: list[dict[str, object]] = []
    async with Session(client=fake_api.client(received)) as session:
        session.begin_step(ADMISSIBLE)
        with pytest.raises(DecisionError, match="request_rejected"):
            await session.decide(state="s", instructions="i", criteria={"look": "look", "jump": "jump"})
    assert received == [] and session.calls == []


async def test_a_decision_sends_one_native_choice_and_returns_its_answer() -> None:
    received: list[dict[str, object]] = []
    async with Session(client=fake_api.client(received)) as session:
        session.begin_step(ADMISSIBLE)
        decision = await session.decide(
            state="Task: look around.", instructions="Pick one.", criteria={c: c for c in ADMISSIBLE}
        )
    assert decision.choice == "go to bed 1" and decision.top_probability == pytest.approx(0.8)
    assert set(decision.probabilities) == set(ADMISSIBLE)
    [payload] = received
    assert payload["model"] == "jev-1.14.0" and payload["state"] == "Task: look around."
    assert payload["questions"] == {
        "decision": {"type": "choice", "instructions": "Pick one.", "criteria": {c: c for c in ADMISSIBLE}}
    }
    assert session.step_decisions[0]["choice"] == "go to bed 1"


async def test_an_answer_from_another_model_version_fails() -> None:
    async with Session(client=fake_api.client(model="jev-9.9.9")) as session:
        session.begin_step(ADMISSIBLE)
        with pytest.raises(DecisionError, match="malformed_answer"):
            await session.decide(state="s", instructions="i", criteria={c: c for c in ADMISSIBLE})
