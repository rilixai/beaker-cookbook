"""Hermetic tests: no network, no AppWorld data, no LLM calls.

They pin the recipe's two contracts:
* the capability-aware model layer attaches exactly the parameters each model
  family supports (acceptance criterion: no 400s from mis-sent params), and
* the runner config keeps the upstream jsonnet semantics (max_steps=50,
  api_predictor with max_predicted_apis=20, Responses API routing).
"""

import json
from pathlib import Path

import pytest
from agents.model_settings import ModelSettings

from appworld_openai_agents.config import PROMPTS_DIR, RunSpec, build_runner_config
from appworld_openai_agents.models import build_model_settings, resolve_profile


RECIPE_DIR = Path(__file__).parent.parent


def test_reasoning_models_omit_sampling_params() -> None:
    profile = resolve_profile("gpt-5.6")
    assert profile.family == "reasoning"
    settings = build_model_settings(profile, reasoning_effort="high", temperature=0.7, seed=1)
    assert settings["reasoning"].effort == "high"
    assert "temperature" not in settings
    assert "top_p" not in settings
    assert "extra_args" not in settings  # no seed
    ModelSettings(**settings)  # accepted by the SDK


def test_standard_models_omit_reasoning() -> None:
    profile = resolve_profile("gpt-4.1")
    assert profile.family == "standard"
    settings = build_model_settings(profile, temperature=0.0, seed=100, reasoning_effort="high")
    assert "reasoning" not in settings
    assert settings["temperature"] == 0.0
    assert settings["extra_args"] == {"seed": 100}
    ModelSettings(**settings)


def test_family_fallbacks_and_overrides() -> None:
    # Unknown snapshot ids route by prefix; an explicit family always wins.
    assert resolve_profile("gpt-5.6-nova-2027-01-01").family == "reasoning"
    assert resolve_profile("gpt-4.2-preview").family == "standard"
    assert resolve_profile("gpt-4.2-preview", family="reasoning").family == "reasoning"


def test_invalid_reasoning_effort_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid reasoning effort"):
        build_model_settings(resolve_profile("gpt-5.6"), reasoning_effort="ultra")


def test_runner_config_preserves_upstream_semantics() -> None:
    config = build_runner_config(RunSpec(model="gpt-5.6", split="dev"))
    assert config["agent"]["max_steps"] == 50
    assert config["agent"]["model"]["type"] == "openai"
    assert config["agent"]["model"]["settings"]["api_type"] == "responses"
    assert config["agent"]["model"]["settings"]["tool_choice"] == "auto"
    assert config["agent"]["model"]["settings"]["parallel_tool_calls"] is True
    assert config["api_predictor"]["mode"] == "predicted"
    assert config["api_predictor"]["max_predicted_apis"] == 20
    assert config["api_predictor"]["demo_task_ids"] == ["82e2fac_1", "29caf6f_1", "d0b1f43_1"]
    assert config["appworld"]["start_servers"] is True
    assert config["dataset"] == "dev"
    for key in ("prompt_file_path", "demo_messages_file_path"):
        assert Path(config["agent"][key]).is_file()
    assert Path(config["api_predictor"]["prompt_file_path"]).is_file()


def test_example_configs_load() -> None:
    reasoning = RunSpec.from_config_file(RECIPE_DIR / "configs" / "gpt-5.6.json")
    assert reasoning.profile.family == "reasoning"
    standard = RunSpec.from_config_file(RECIPE_DIR / "configs" / "gpt-4.1.json")
    assert standard.profile.family == "standard"
    # CLI overrides win over the file.
    overridden = RunSpec.from_config_file(
        RECIPE_DIR / "configs" / "gpt-5.6.json", reasoning_effort="low", split="test_normal"
    )
    assert overridden.reasoning_effort == "low"
    assert overridden.split == "test_normal"


def test_vendored_prompts_match_expected_shape() -> None:
    demos = json.loads((PROMPTS_DIR / "function_calling_agent" / "demos.json").read_text())
    assert isinstance(demos, list) and demos, "demos.json should be a non-empty message list"
    instructions = (PROMPTS_DIR / "function_calling_agent" / "instructions.txt").read_text()
    assert "{max_steps}" in instructions
