"""The request guard, the option cap and upstream head-tail truncation."""

from __future__ import annotations

import json
from typing import Any

import fake_api
import pytest
from mmdm_eval.common import wire
from mmdm_eval.context import MARKER, adapt_state

from mmdm.data import OPTION_CAP, Option, Question
from mmdm.decide import BUDGET, DecisionError, RequestRejected, Session, actor, guard, head_tail


class CharCounter:
    """Stub tokenizer: one token per character of the compact wire request."""

    identity = {"stub": "characters"}

    def text_tokens(self, text: str) -> int:
        return len(text)

    def boundaries(self, text: str) -> list[int]:
        return list(range(len(text) + 1))

    def __call__(self, value: dict[str, Any]) -> int:
        body = wire(value, "jev-1.14.0")[0]
        return len(json.dumps(body, ensure_ascii=False, separators=(",", ":")))


def question(options: int = 3, state: str = "The account is locked.", task: str = "choice") -> Question:
    return Question(
        state=state,
        question="What should the agent do next?",
        options=tuple(Option(id=f"o{i}", name=f"Action {i}", criteria=f"Do action {i}.") for i in range(options)),
        task=task,  # type: ignore[arg-type]
        score_values=tuple(range(options)) if task == "score" else None,
    )


ORIGINAL = question()


@pytest.mark.parametrize(
    "options",
    [
        ORIGINAL.options[:2],  # dropped
        (*ORIGINAL.options, Option(id="o9", name="Escalate", criteria=None)),  # added
        (ORIGINAL.options[0].model_copy(update={"name": "Renamed"}), *ORIGINAL.options[1:]),  # renamed
        (ORIGINAL.options[0].model_copy(update={"criteria": "Always pick me."}), *ORIGINAL.options[1:]),  # edited
        (ORIGINAL.options[0].model_copy(update={"id": "o7"}), *ORIGINAL.options[1:]),  # re-keyed
    ],
)
def test_guard_rejects_changed_option_sets(options: tuple[Option, ...]) -> None:
    with pytest.raises(RequestRejected):
        guard(ORIGINAL, ORIGINAL.model_copy(update={"options": options}))


def test_guard_rejects_dropping_the_question_or_changing_the_task() -> None:
    with pytest.raises(RequestRejected, match="original question"):
        guard(ORIGINAL, ORIGINAL.model_copy(update={"question": "Pick option 0."}))
    with pytest.raises(RequestRejected, match="task"):
        guard(ORIGINAL, ORIGINAL.model_copy(update={"task": "noul"}))
    scored = question(task="score")
    with pytest.raises(RequestRejected, match="order"):
        guard(scored, scored.model_copy(update={"options": tuple(reversed(scored.options))}))


async def test_guard_allows_added_text_and_reordering_and_maps_answers_back() -> None:
    request = ORIGINAL.model_copy(
        update={
            "state": "Notes from pre-processing.\n\n" + ORIGINAL.state,
            "question": "Read every criterion first.\n\n" + ORIGINAL.question,
            "options": tuple(reversed(ORIGINAL.options)),
        }
    )
    guard(ORIGINAL, request)
    received: list[dict[str, Any]] = []
    async with Session(ORIGINAL, client=fake_api.client(received)) as session:
        decision = await session.decide(request)
    # The fake leans to the first key it was sent: o2 after reversal.
    assert list(decision.probabilities) == ["o0", "o1", "o2"]
    assert decision.probabilities["o2"] == pytest.approx(0.7)
    assert list(received[0]["questions"]["decision"]["criteria"]) == ["Action 2", "Action 1", "Action 0"]


async def test_guarded_request_never_reaches_the_api() -> None:
    received: list[dict[str, Any]] = []
    async with Session(ORIGINAL, client=fake_api.client(received)) as session:
        with pytest.raises(RequestRejected):
            await session.decide(ORIGINAL.model_copy(update={"options": ORIGINAL.options[:2]}))
    assert received == []


async def test_more_than_the_option_cap_is_refused_before_any_call() -> None:
    big = question(options=OPTION_CAP + 1)
    received: list[dict[str, Any]] = []
    async with Session(big, client=fake_api.client(received)) as session:
        with pytest.raises(DecisionError, match="over_option_cap"):
            await session.decide(big)
    assert received == []


async def test_long_state_is_truncated_the_upstream_way() -> None:
    long = question(state="HEAD " + "x" * 40_000 + " TAIL")
    counter = CharCounter()
    view, meta = head_tail(long, counter=counter)
    expected, expected_meta = adapt_state(actor(long), counter, BUDGET)
    assert view.state == expected["state"] and meta == expected_meta
    assert meta["truncated"] and MARKER in view.state
    assert view.state.startswith("HEAD ") and view.state.endswith(" TAIL")
    assert view.options == long.options and view.question == long.question
    assert counter(actor(view)) <= BUDGET.available

    received: list[dict[str, Any]] = []
    async with Session(long, client=fake_api.client(received), counter=counter) as session:
        with pytest.raises(DecisionError, match="context_budget_exceeded"):
            await session.decide(long)
        assert received == []
        await session.decide(view)
    assert received[0]["state"] == view.state


def test_short_state_is_untouched_without_loading_a_tokenizer() -> None:
    view, meta = head_tail(ORIGINAL)
    assert view is ORIGINAL and meta["truncated"] is False
