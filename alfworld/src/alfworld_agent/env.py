"""Load one ALFWorld game, step it, and enforce the rules. Fixed: not part of the editable surface.

The game is built the way upstream ``AlfredTWEnv.init_env`` builds it
(``textworld.gym`` with the ``AlfredDemangler`` and ``AlfredInfos`` wrappers),
but for one game file and without the expert wrapper. Only
``admissible_commands`` and ``won`` are requested from TextWorld, and ``won``
stays here: the agent sees a ``View`` with the goal, the current observation,
its own history and the admissible commands, nothing else.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import textworld
import textworld.gym
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

from alfworld_agent.decide import CallRecord, DecisionError, Session, Tracer


# The usual episode limit in LLM-agent papers on ALFWorld.
STEP_LIMIT = 50
GOAL_MARKER = "Your task is to: "
# TextWorld keeps module-level state that is not thread-safe: the PDDL translator
# (loading) and shared TatSu parsers (``textworld.logic`` and the PDDL text
# generator, used on every reset and step). Games in one process load and step
# one at a time; a step takes milliseconds, so this costs little next to model calls.
_ENGINE_LOCK = threading.Lock()


class EnvError(RuntimeError):
    """The game failed to load or step."""


class InadmissibleCommand(ValueError):
    """The agent sent a command that is not currently admissible."""


@dataclass(frozen=True)
class Step:
    action: str
    observation: str


@dataclass(frozen=True)
class View:
    """Everything the agent sees before one action."""

    goal: str
    observation: str
    history: tuple[Step, ...]
    admissible_commands: tuple[str, ...]
    step: int
    step_limit: int = STEP_LIMIT


def split_goal(text: str) -> tuple[str, str]:
    """The first observation is a banner, the room, then ``Your task is to: <goal>``."""
    room, marker, goal = text.partition(GOAL_MARKER)
    if not marker or not goal.strip():
        raise EnvError("The first observation has no task goal.")
    lines = [line for line in room.strip().splitlines() if not line.startswith("-=")]
    return goal.strip(), "\n".join(lines).strip()


class Game:
    """One loaded game. Holds the path and ``won``; hands the agent only a ``View``."""

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise EnvError(f"Game file not found: {path.name}")
        infos = textworld.EnvInfos(won=True, admissible_commands=True)
        try:
            with _ENGINE_LOCK:
                env_id = textworld.gym.register_game(
                    str(path),
                    infos,
                    max_episode_steps=STEP_LIMIT,
                    wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
                )
                self._env = textworld.gym.make(env_id)
                text, state = self._env.reset()
        except Exception as exc:
            raise EnvError(f"{type(exc).__name__}: {exc}"[:300]) from exc
        self.goal, self.observation = split_goal(text)
        self._admissible = tuple(state["admissible_commands"])
        self._won = bool(state["won"])
        self.history: list[Step] = []

    @property
    def won(self) -> bool:
        return self._won

    @property
    def steps(self) -> int:
        return len(self.history)

    @property
    def done(self) -> bool:
        return self._won or self.steps >= STEP_LIMIT

    def view(self) -> View:
        return View(
            goal=self.goal,
            observation=self.observation,
            history=tuple(self.history),
            admissible_commands=self._admissible,
            step=self.steps,
        )

    def step(self, command: str) -> None:
        if self.done:
            raise EnvError("The episode is over.")
        if not isinstance(command, str) or command not in self._admissible:
            raise InadmissibleCommand(repr(command)[:200])
        try:
            with _ENGINE_LOCK:
                text, _, _, state = self._env.step(command)
        except Exception as exc:
            raise EnvError(f"{type(exc).__name__}: {exc}"[:300]) from exc
        self.observation = text.strip()
        self.history.append(Step(command, self.observation))
        self._admissible = tuple(state["admissible_commands"])
        self._won = bool(state["won"])

    def close(self) -> None:
        with _ENGINE_LOCK:
            self._env.close()


Act = Callable[[View, Session], Awaitable[str]]


def count_repeats(history: list[Step]) -> int:
    """Steps that repeat the previous step's action and observation: a loop detector."""
    return sum(1 for previous, current in zip(history, history[1:], strict=False) if previous == current)


@dataclass
class Episode:
    case_id: str
    goal: str = ""
    won: bool = False
    steps: int = 0
    repeats: int = 0
    error: str | None = None
    retryable: bool = False
    transcript: list[dict[str, Any]] = field(default_factory=list)
    calls: list[CallRecord] = field(default_factory=list)
    latency_s: float = 0.0

    @property
    def hit_step_limit(self) -> bool:
        return not self.won and self.steps >= STEP_LIMIT

    def output(self, last: int = 3) -> dict[str, Any]:
        """The displayed result: goal, last few steps, outcome. Scoring reads only this."""
        decisions = [c for c in self.calls if c.operation == "systemone"]
        tops = [c.top_probability for c in decisions if c.top_probability is not None]
        return {
            "goal": self.goal,
            "last_steps": [
                {"step": s["step"], "action": s.get("action"), "observation": s.get("result")}
                for s in self.transcript[-last:]
            ],
            "outcome": {
                "won": self.won,
                "steps": self.steps,
                "hit_step_limit": self.hit_step_limit,
                "error": self.error,
            },
            "decision_calls": len(decisions),
            "chat_calls": len(self.calls) - len(decisions),
            "repeats": self.repeats,
            "mean_top_probability": sum(tops) / len(tops) if tops else None,
            "decision_input_tokens": sum(c.input_tokens or 0 for c in decisions),
            "decision_output_tokens": sum(c.output_tokens or 0 for c in decisions),
            "models": sorted({c.model for c in self.calls}),
            "latency_s": self.latency_s,
        }

    def context(self) -> dict[str, Any]:
        """The full transcript: each step's observation, options, probabilities and chosen action."""
        return {
            "transcript": self.transcript,
            "decision_latencies_s": [c.latency_s for c in self.calls if c.operation == "systemone"],
        }


async def run_episode(case_id: str, game_path: Path, *, act: Act, session: Session, trace: Tracer) -> Episode:
    """Play one game until won or the step limit. Every failure is returned with an observable error."""
    episode = Episode(case_id)
    start = time.monotonic()
    game: Game | None = None
    with trace.stage("alfworld.episode", inputs={"id": case_id}) as episode_stage:
        try:
            game = await asyncio.to_thread(Game, game_path)
            episode.goal = game.goal
            while not game.done:
                view = game.view()
                session.begin_step(view.admissible_commands)
                record: dict[str, Any] = {
                    "step": view.step + 1,
                    "observation": view.observation,
                    "options": list(view.admissible_commands),
                }
                episode.transcript.append(record)
                with trace.stage(
                    "alfworld.step", inputs={"step": view.step + 1, "option_count": len(view.admissible_commands)}
                ) as stage:
                    try:
                        command = await act(view, session)
                    finally:
                        record["decisions"] = session.step_decisions
                    record["action"] = command
                    await asyncio.to_thread(game.step, command)
                    record["result"] = game.observation
                    stage.output({"action": command, "observation": game.observation})
        except DecisionError as exc:
            episode.error, episode.retryable = str(exc), exc.retryable
        except InadmissibleCommand as exc:
            episode.error = f"inadmissible_command: {exc}"
        except EnvError as exc:
            episode.error = f"env_error: {exc}"
        except Exception as exc:  # a candidate bug is a failed case, not a crashed run
            episode.error = f"agent_error: {type(exc).__name__}: {exc}"[:500]
        finally:
            if game is not None:
                episode.won, episode.steps, episode.repeats = game.won, game.steps, count_repeats(game.history)
                game.close()
            episode.calls = list(session.calls)
            episode.latency_s = time.monotonic() - start
        episode_stage.output(episode.output()["outcome"])
    return episode
