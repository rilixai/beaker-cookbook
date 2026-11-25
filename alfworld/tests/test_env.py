"""The environment on one small real game: loading, the admissibility rule, what the agent sees, the loop."""

from __future__ import annotations

import dataclasses
import json

import fake_api
import pytest

from alfworld_agent import agent
from alfworld_agent.decide import Session
from alfworld_agent.env import STEP_LIMIT, Game, InadmissibleCommand, View, run_episode
from alfworld_agent.scoring import score


def test_loads_one_game_and_lists_admissible_commands() -> None:
    game = Game(fake_api.game_path())
    try:
        view = game.view()
        assert view.goal == "put a spraybottle in garbagecan."
        assert view.observation.startswith("You are in the middle of a room.")
        assert "Welcome" not in view.observation and "Your task is to" not in view.observation
        assert view.step == 0 and view.history == () and view.step_limit == STEP_LIMIT
        assert "go to countertop 1" in view.admissible_commands and "look" in view.admissible_commands
    finally:
        game.close()


def test_rejects_an_inadmissible_command() -> None:
    game = Game(fake_api.game_path())
    try:
        before = game.view()
        with pytest.raises(InadmissibleCommand):
            game.step("take spraybottle 2 from countertop 1")  # not admissible before going there
        assert game.view() == before
        game.step("go to countertop 1")
        assert game.steps == 1 and "take spraybottle 2 from countertop 1" in game.view().admissible_commands
    finally:
        game.close()


def test_the_agent_never_sees_won_facts_expert_plan_task_type_or_game_path() -> None:
    assert [f.name for f in dataclasses.fields(View)] == [
        "goal",
        "observation",
        "history",
        "admissible_commands",
        "step",
        "step_limit",
    ]
    game = Game(fake_api.game_path())
    try:
        game.step(fake_api.walkthrough()[0])
        seen = json.dumps(dataclasses.asdict(game.view()))
    finally:
        game.close()
    for leak in (
        "tw-pddl",
        "valid_seen",
        "trial_T",
        "pick_and_place_simple",
        "walkthrough",
        '"won"',
        "SprayBottle-None",
    ):
        assert leak not in seen
    # Only these infos are requested from TextWorld; facts and the expert plan never are.
    assert sorted(game._env.request_infos.basics) == ["admissible_commands", "won"]
    assert game._env.request_infos.extras == []


async def test_a_scripted_decider_replays_the_walkthrough_and_wins() -> None:
    received: list[dict[str, object]] = []
    client = fake_api.client(received, choose=fake_api.scripted(fake_api.walkthrough()))
    policy = agent.Agent("Choose the next action that makes progress toward the task goal.")
    async with Session(client=client) as session:
        episode = await run_episode("case", fake_api.game_path(), act=policy.act, session=session, trace=session.trace)
    assert episode.error is None and episode.won and episode.steps == 4
    assert len(received) == 4 and all(p["model"] == "jev-1.14.0" for p in received)
    assert [s["action"] for s in episode.transcript] == fake_api.walkthrough()
    assert all(len(s["decisions"]) == 1 and s["options"] for s in episode.transcript)

    scored = score(episode.output())
    assert scored.objective == 1.0 and scored.success and not scored.hit_step_limit
    assert scored.field_scores == {
        "success": 1.0,
        "steps": 4 / STEP_LIMIT,
        "decision_calls": 4 / (2 * STEP_LIMIT),
        "repeat_rate": 0.0,
        "mean_top_probability": pytest.approx(0.8),
    }


async def test_the_step_limit_stops_the_episode_at_fifty() -> None:
    async def look(view: View, session: Session) -> str:
        return "look"

    async with Session() as session:
        episode = await run_episode("case", fake_api.game_path(), act=look, session=session, trace=session.trace)
    assert episode.error is None and not episode.won and episode.steps == STEP_LIMIT
    assert episode.hit_step_limit and episode.repeats == STEP_LIMIT - 1
    scored = score(episode.output())
    assert scored.objective == 0.0 and scored.hit_step_limit
    assert scored.field_scores["steps"] == 1.0 and scored.field_scores["repeat_rate"] == (STEP_LIMIT - 1) / STEP_LIMIT


async def test_an_inadmissible_command_fails_the_case_with_an_error() -> None:
    async def invent(view: View, session: Session) -> str:
        return "fly to the moon"

    async with Session() as session:
        episode = await run_episode("case", fake_api.game_path(), act=invent, session=session, trace=session.trace)
    assert episode.error == "inadmissible_command: 'fly to the moon'" and episode.steps == 0
    scored = score(episode.output())
    assert scored.objective == 0.0 and scored.error == episode.error
