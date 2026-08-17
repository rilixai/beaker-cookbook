"""Assembles the AppWorld ``openai_agents`` runner config and executes it.

This is the Python translation of upstream's reference jsonnet config
(``experiments/configs/openai_agents_mcp_agent/openai/.../*.jsonnet`` at the
vendored commit), with the model block supplied by a capability-aware
:class:`~appworld_openai_agents.models.ModelProfile` instead of being
hardcoded. The agent-loop semantics (max_steps=50, the api_predictor pass,
server setup) are preserved; deliberate deviations from the reference config
are listed in ATTRIBUTION.md.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from appworld_openai_agents.models import ModelProfile
from appworld_openai_agents.vendored.openai_agents.run import run_agent_on_tasks


PROMPTS_DIR = Path(__file__).parent / "prompts"

# Upstream reference config values (openai_agents_mcp_agent).
MAX_STEPS = 50
MAX_PREDICTED_APIS = 20
API_PREDICTOR_DEMO_TASK_IDS = ["82e2fac_1", "29caf6f_1", "d0b1f43_1"]
RANDOM_SEED = 100

# How the agent's tool list is chosen before the loop starts:
# - "predicted": an LLM pre-pass picks the <=20 APIs the task likely needs
#   (upstream's reference baseline)
# - "all": no pre-fill — all 457 APIs are exposed as tools (harder)
# - "ground_truth": oracle tool list from the task's ground truth (easier)
API_MODES = ("predicted", "all", "ground_truth")


def build_runner_config(
    profile: ModelProfile, max_steps: int = MAX_STEPS, api_mode: str = "predicted"
) -> dict[str, Any]:
    if api_mode not in API_MODES:
        raise ValueError(f"Unknown api_mode {api_mode!r}; expected one of {API_MODES}.")
    return {
        "agent": {
            "model": profile.to_model_config(for_agent=True),
            "max_steps": max_steps,
            "prompt_file_path": str(PROMPTS_DIR / "function_calling_agent" / "instructions.txt"),
            "demo_messages_file_path": str(PROMPTS_DIR / "function_calling_agent" / "demos.json"),
        },
        "api_predictor": {
            # The "pick which APIs become tools" pass that runs before the
            # agent loop (upstream default: mode=predicted, max_predicted_apis=20).
            "mode": api_mode,
            "model_config": profile.to_model_config(for_agent=False),
            "prompt_file_path": str(PROMPTS_DIR / "api_predictor.txt"),
            "demo_task_ids": API_PREDICTOR_DEMO_TASK_IDS,
            # Upstream's predictor truncates the API list to max_predicted_apis
            # in every mode, so lift the cap for the non-predicted modes
            # ("all" must expose all 457 APIs, "ground_truth" the full oracle
            # list).
            "max_predicted_apis": MAX_PREDICTED_APIS if api_mode == "predicted" else 10**9,
        },
        "appworld": {
            "start_servers": True,
            "show_server_logs": False,
            "remote_apis_url": "http://localhost:{port}",
            "remote_mcp_url": "http://localhost:{port}",
            "mcp_server_kwargs": {"output_type": "both_but_empty_text"},
            "random_seed": RANDOM_SEED,
            "include_direct_functions": True,
            "direct_function_separator": "__",
        },
        "logger": {"color": True, "verbose": True},
    }


def run(
    experiment_name: str,
    task_ids: list[str],
    profile: ModelProfile,
    max_steps: int = MAX_STEPS,
    api_mode: str = "predicted",
) -> None:
    """Run the vendored openai_agents agent (api_predictor included) over
    ``task_ids``, writing predictions under
    ``$APPWORLD_ROOT/experiments/outputs/{experiment_name}`` in the format
    ``appworld evaluate`` expects."""
    config = build_runner_config(profile, max_steps=max_steps, api_mode=api_mode)
    asyncio.run(
        run_agent_on_tasks(
            experiment_name=experiment_name,
            task_ids=task_ids,
            api_predictor_config=config["api_predictor"],
            agent_config=config["agent"],
            appworld_config=config["appworld"],
            logger_config=config["logger"],
        )
    )
