"""Entry point that executes the code agent over a list of tasks.

The agent-loop semantics mirror AppWorld's ReAct baseline (max_steps=50,
random_seed=100, per-task worlds saved for the evaluator); the loop itself is
implemented in :mod:`appworld_openai_agents_sdk.code_agent` on the OpenAI Agents
SDK.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from agents import OpenAIChatCompletionsModel
from agents.run import RunConfig
from openai import AsyncOpenAI

from appworld_openai_agents_sdk.code_agent import run_code_agent_on_tasks
from appworld_openai_agents_sdk.models import ModelProfile


PROMPTS_DIR = Path(__file__).parent / "prompts"

# Upstream reference values (react_code_agent).
MAX_STEPS = 50
RANDOM_SEED = 100


def run(
    experiment_name: str,
    task_ids: list[str],
    profile: ModelProfile,
    max_steps: int = MAX_STEPS,
) -> None:
    """Run the code-execution ReAct agent (no tool pre-selection; API
    discovery via api_docs) over ``task_ids``, writing predictions under
    ``$APPWORLD_ROOT/experiments/outputs/{experiment_name}`` in the format
    ``appworld evaluate`` expects."""

    async def run_tasks() -> None:
        client = endpoint_client(profile) if profile.base_url else None
        try:
            await run_code_agent_on_tasks(
                experiment_name=experiment_name,
                task_ids=task_ids,
                profile=profile,
                prompt_file_path=str(PROMPTS_DIR / "react_code_agent" / "instructions.txt"),
                appworld_config={"random_seed": RANDOM_SEED},
                logger_config={"color": True, "verbose": True},
                max_steps=max_steps,
                run_config=endpoint_run_config(profile, client) if client else None,
            )
        finally:
            if client is not None:
                await client.close()

    asyncio.run(run_tasks(), loop_factory=RealClockLoop)


class RealClockLoop(asyncio.SelectorEventLoop):
    """AppWorld freezes Python's clocks inside a task. Request timeouts and the
    OpenAI client's retry backoff must use real elapsed time, or a retried
    request waits forever. The hosted case worker uses the same loop."""

    def time(self) -> float:
        return time.clock_gettime(time.CLOCK_MONOTONIC)


def endpoint_client(profile: ModelProfile) -> AsyncOpenAI:
    """A client for the profile's OpenAI-compatible endpoint."""
    assert profile.base_url is not None and profile.api_key_env is not None
    key = os.environ.get(profile.api_key_env)
    if not key:
        raise RuntimeError(f"Set {profile.api_key_env} to use {profile.base_url} ({profile.name}).")
    return AsyncOpenAI(base_url=profile.base_url, api_key=key)


def endpoint_run_config(profile: ModelProfile, client: AsyncOpenAI) -> RunConfig:
    """Send the agent's model calls to the endpoint as Chat Completions."""
    return RunConfig(
        model=OpenAIChatCompletionsModel(model=profile.api_model, openai_client=client),
        tracing_disabled=True,
    )
