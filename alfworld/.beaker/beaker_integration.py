"""Optimize an ALFWorld decision agent; keep the games, the rules, decision calls and scoring fixed.

Editable: ``prompts/`` and ``src/alfworld_agent/agent.py``. Fixed: the game
files, ``env.py`` (loading, stepping, the admissibility rule and the 50-step
limit), ``decide.py`` (the request guard and every traced model call),
``data.py`` and ``scoring.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from beaker import (
    Case,
    CaseResult,
    CaseScore,
    Check,
    Integration,
    JsonValue,
    RepositoryRunSetupResult,
    RetryableCaseError,
    RolloutRuntime,
    RunSetup,
    RunSetupResults,
    SetupRuntime,
    inference_target,
    repository,
)
from openai import AsyncOpenAI
from pydantic import ConfigDict, Field

from alfworld_agent import agent
from alfworld_agent.data import GameRef, Row, games_root
from alfworld_agent.decide import (
    CHAT_TIMEOUT_SECONDS,
    DEFAULT_TARGET,
    ChatModel,
    DecisionTarget,
    Session,
    default_chat_model,
)
from alfworld_agent.env import STEP_LIMIT, run_episode
from alfworld_agent.scoring import score


class CaseInput(GameRef):
    """The game plus its case ID, which names the trace stage. The agent sees neither."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(min_length=1)


class Setup(RunSetup[Row]):
    row_model = Row

    @asynccontextmanager
    async def prepare_run(self, *, runtime: SetupRuntime) -> AsyncIterator[RunSetupResults]:
        del runtime
        # Downloads and verifies the pinned game files once per machine.
        self.root = await asyncio.to_thread(games_root)
        yield {"repository": RepositoryRunSetupResult()}

    async def load_cases(self, row: Row, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        del runtime
        if not (self.root / row.input.game).is_file():
            raise ValueError(f"Game file missing for {row.id}")
        yield Case(
            id=row.id,
            input={"id": row.id, **row.input.model_dump(mode="json")},
            expected=row.expected.model_dump(mode="json"),
        )


def _decision_target(runtime: RolloutRuntime[Any]) -> DecisionTarget:
    """The run's selected decision model through Beaker's gateway; else native Jev."""
    if runtime.model:
        target = inference_target(runtime)
        return DecisionTarget(
            model=target.model,
            provider=target.model.partition(":")[0] or DEFAULT_TARGET.provider,
            server_address=urlparse(target.base_url).hostname or target.base_url,
            base_url=target.base_url,
            api_key=target.api_key,
        )
    return DEFAULT_TARGET


def _client_for(runtime: RolloutRuntime[Any]) -> ChatModel:
    """For chat calls a candidate adds: the run's selected model through the gateway; else the pinned default.

    A model choice resolves to a gateway endpoint, a run-scoped token and a
    canonical ``<provider>:<model>`` name; the gateway applies the run's
    reasoning effort, so none is sent.
    """
    if runtime.model:
        target = inference_target(runtime)
        client = AsyncOpenAI(api_key=target.api_key, base_url=target.base_url, timeout=CHAT_TIMEOUT_SECONDS)
        return ChatModel(client=client, provider=target.model.partition(":")[0], model=target.model)
    return default_chat_model()


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime[Any]) -> CaseResult:
    case = CaseInput.model_validate(case_input)
    root = await asyncio.to_thread(games_root)
    # Repository candidates execute with source_dir as their working directory.
    policy = agent.Agent(agent.load_instructions(Path.cwd() / "prompts"))
    async with Session(
        target=_decision_target(runtime), trace=runtime.trace, chat_factory=lambda: _client_for(runtime)
    ) as session:
        episode = await run_episode(case.id, root / case.game, act=policy.act, session=session, trace=runtime.trace)
    if episode.retryable:
        raise RetryableCaseError(episode.error or "transient decision API error")
    return CaseResult(output=episode.output(), output_kind="record", context=episode.context())


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    del case, case_files_dir
    scored = score(result.output)
    return CaseScore(
        objective=scored.objective,
        field_scores=scored.field_scores,
        checks=(
            Check(
                name="success",
                verdict="pass" if scored.success else "fail",
                message=scored.error,
            ),
            Check(
                name="step_limit",
                verdict="fail" if scored.hit_step_limit else "pass",
                message=f"Used all {STEP_LIMIT} steps without finishing the task." if scored.hit_step_limit else None,
            ),
        ),
    )


integration = Integration(
    targets=repository(paths=("prompts", "src/alfworld_agent/agent.py")),
    run_setup=Setup,
    run_case=run_case,
    score_case=score_case,
)
