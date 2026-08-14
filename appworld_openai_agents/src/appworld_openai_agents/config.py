"""Runner-config assembly.

Translates the upstream jsonnet experiment config
(``experiments/configs/openai_agents_mcp_agent/openai/.../*.jsonnet`` at the
vendored commit) into a plain Python dict for the vendored ``run.py``,
preserving its semantics: ``max_steps=50``, the ``api_predictor``
("retrieve relevant APIs first") pass with ``max_predicted_apis=20``, the
same demo task ids, and the same AppWorld server/MCP setup — while swapping
the fixed ``gpt-4o`` model block for the capability-aware layer in
:mod:`appworld_openai_agents.models`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import ModelFamily, ModelProfile, build_model_settings, resolve_profile


PROMPTS_DIR = Path(__file__).parent / "prompts"

# Upstream defaults (see the jsonnet reference config).
DEFAULT_MAX_STEPS = 50
DEFAULT_MAX_PREDICTED_APIS = 20
DEFAULT_API_PREDICTOR_DEMO_TASK_IDS = ["82e2fac_1", "29caf6f_1", "d0b1f43_1"]
DEFAULT_RANDOM_SEED = 100


@dataclass
class RunSpec:
    """Everything one experiment run needs, resolved from config file + CLI flags."""

    model: str
    family: ModelFamily | None = None
    reasoning_effort: str = "medium"
    temperature: float = 0.0
    top_p: float | None = None
    seed: int | None = DEFAULT_RANDOM_SEED
    max_output_tokens: int | None = 65536
    max_steps: int = DEFAULT_MAX_STEPS
    max_predicted_apis: int = DEFAULT_MAX_PREDICTED_APIS
    split: str = "dev"
    verbose: bool = True

    @property
    def profile(self) -> ModelProfile:
        return resolve_profile(self.model, self.family)

    @classmethod
    def from_config_file(cls, path: str | Path, **overrides: Any) -> "RunSpec":
        """Load a JSON model config (see ``configs/``) and apply CLI overrides."""
        data = json.loads(Path(path).read_text())
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)

    def experiment_name(self) -> str:
        model_slug = self.model.replace("/", "-").replace(".", "-")
        if self.profile.family == "reasoning":
            model_slug += f"-{self.reasoning_effort}"
        return f"openai_agents_{model_slug}_{self.split}"


def build_runner_config(spec: RunSpec) -> dict[str, Any]:
    """The runner config consumed by the vendored ``run.run_experiment``."""
    profile = spec.profile
    settings = build_model_settings(
        profile,
        reasoning_effort=spec.reasoning_effort,
        temperature=spec.temperature,
        top_p=spec.top_p,
        seed=spec.seed,
        max_output_tokens=spec.max_output_tokens,
    )
    # `type: openai` + `api_type: responses` routes to the Agents SDK's native
    # OpenAI Responses model (NOT LitellmModel), so reasoning settings are honored.
    agent_model = {
        "type": "openai",
        "name": spec.model,
        "settings": {
            **settings,
            "api_type": "responses",
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        },
    }
    predictor_model = {
        "type": "openai",
        "name": spec.model,
        "api_type": "responses",
        "settings": settings,
    }
    return {
        "agent": {
            "model": agent_model,
            "max_steps": spec.max_steps,
            "prompt_file_path": str(PROMPTS_DIR / "function_calling_agent" / "instructions.txt"),
            "demo_messages_file_path": str(PROMPTS_DIR / "function_calling_agent" / "demos.json"),
        },
        "api_predictor": {
            "mode": "predicted",
            "model_config": predictor_model,
            "prompt_file_path": str(PROMPTS_DIR / "api_predictor.txt"),
            "demo_task_ids": list(DEFAULT_API_PREDICTOR_DEMO_TASK_IDS),
            "max_predicted_apis": spec.max_predicted_apis,
        },
        "appworld": {
            "start_servers": True,
            "show_server_logs": False,
            "remote_apis_url": "http://localhost:{port}",
            "remote_mcp_url": "http://localhost:{port}",
            "mcp_server_kwargs": {
                "output_type": "both_but_empty_text",
            },
            "random_seed": DEFAULT_RANDOM_SEED,
            "include_direct_functions": True,
            "direct_function_separator": "__",
        },
        "logger": {
            "color": True,
            "verbose": spec.verbose,
        },
        "dataset": spec.split,
    }
