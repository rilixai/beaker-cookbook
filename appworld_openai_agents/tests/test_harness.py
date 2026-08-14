"""Hermetic structural checks — no network, no API keys, no AppWorld data.

They verify the property this recipe exists for: that the capability layer
attaches exactly the right parameters per model family, and that the vendored
scaffold + prompts + example configs are wired up and importable.
"""

from pathlib import Path

import pytest
from agents.model_settings import ModelSettings

from appworld_openai_agents.models import DEFAULT_MAX_OUTPUT_TOKENS, ModelProfile
from appworld_openai_agents.runner import PROMPTS_DIR, build_runner_config


RECIPE_DIR = Path(__file__).parent.parent


def _model_settings_kwargs(settings: dict) -> dict:
    return {k: v for k, v in settings.items() if k not in ("api_type", "base_url", "api_key_env_name")}


def test_reasoning_profile_omits_sampling_params() -> None:
    settings = ModelProfile(name="gpt-5.6", family="reasoning", reasoning_effort="high").settings()
    assert "temperature" not in settings
    assert "top_p" not in settings
    assert "seed" not in settings
    assert settings["reasoning"].effort == "high"
    assert settings["api_type"] == "responses"
    # The dict must be consumable by the SDK's ModelSettings once the runner
    # pops the routing keys.
    ModelSettings(**_model_settings_kwargs(settings))


def test_standard_profile_omits_reasoning() -> None:
    settings = ModelProfile(name="gpt-4.1", family="standard", temperature=0.3).settings()
    assert "reasoning" not in settings
    assert settings["temperature"] == 0.3
    ModelSettings(**_model_settings_kwargs(settings))


def test_family_default_output_budgets() -> None:
    assert DEFAULT_MAX_OUTPUT_TOKENS["reasoning"] > DEFAULT_MAX_OUTPUT_TOKENS["standard"]
    profile = ModelProfile(name="gpt-5.6", family="reasoning", max_output_tokens=1234)
    assert profile.settings()["max_tokens"] == 1234


def test_invalid_effort_rejected() -> None:
    with pytest.raises(ValueError):
        ModelProfile(name="gpt-5.6", family="reasoning", reasoning_effort="ultra")


def test_predictor_settings_have_no_routing_keys() -> None:
    settings = ModelProfile(name="gpt-5.6", family="reasoning").settings(for_agent=False)
    assert "api_type" not in settings
    assert "tool_choice" not in settings
    ModelSettings(**settings)


def test_example_configs_load() -> None:
    reasoning = ModelProfile.from_toml(RECIPE_DIR / "configs" / "gpt-5.6.toml")
    assert reasoning.family == "reasoning" and reasoning.reasoning_effort == "medium"
    standard = ModelProfile.from_toml(RECIPE_DIR / "configs" / "gpt-4.1.toml")
    assert standard.family == "standard" and standard.temperature == 0.0


def test_runner_config_matches_upstream_semantics() -> None:
    config = build_runner_config(ModelProfile(name="gpt-4.1", family="standard"))
    agent = config["agent"]
    assert agent["max_steps"] == 50
    assert agent["model"]["type"] == "openai"
    assert agent["model"]["settings"]["tool_choice"] == "auto"
    assert agent["model"]["settings"]["parallel_tool_calls"] is True
    predictor = config["api_predictor"]
    assert predictor["mode"] == "predicted"
    assert predictor["max_predicted_apis"] == 20
    assert config["appworld"]["start_servers"] is True
    for key in ("prompt_file_path", "demo_messages_file_path"):
        assert Path(agent[key]).is_file()
    assert Path(predictor["prompt_file_path"]).is_file()
    assert (PROMPTS_DIR / "function_calling_agent" / "demos.json").is_file()


def test_vendored_scaffold_importable_and_self_contained() -> None:
    from appworld_openai_agents.vendored.openai_agents import run as vendored_run

    assert hasattr(vendored_run, "run_agent_on_tasks")
    vendored_dir = RECIPE_DIR / "src" / "appworld_openai_agents" / "vendored"
    for path in vendored_dir.rglob("*.py"):
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                assert "appworld_agents" not in stripped, f"{path} still imports the un-packaged upstream tree"
