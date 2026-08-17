"""Entry point that executes the code agent over a list of tasks.

The agent-loop semantics mirror AppWorld's ReAct baseline (max_steps=50,
random_seed=100, per-task worlds saved for the evaluator); the loop itself is
implemented in :mod:`appworld_openai_agents_sdk.code_agent` on the OpenAI Agents
SDK.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

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
    asyncio.run(
        run_code_agent_on_tasks(
            experiment_name=experiment_name,
            task_ids=task_ids,
            profile=profile,
            prompt_file_path=str(PROMPTS_DIR / "react_code_agent" / "instructions.txt"),
            appworld_config={"random_seed": RANDOM_SEED},
            logger_config={"color": True, "verbose": True},
            max_steps=max_steps,
        )
    )
