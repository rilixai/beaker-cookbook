"""Hermetic structural checks — no network, no API keys, no AppWorld data.

They verify the property this recipe exists for: that the capability layer
attaches exactly the right parameters per model family, and that the agent,
prompt, and example configs are wired up and importable.
"""

from pathlib import Path

import pytest
from agents.model_settings import ModelSettings

from appworld_openai_agents_sdk.cli import _parse_args, _profile_from_args
from appworld_openai_agents_sdk.models import DEFAULT_MAX_OUTPUT_TOKENS, ModelProfile
from appworld_openai_agents_sdk.runner import MAX_STEPS, PROMPTS_DIR


RECIPE_DIR = Path(__file__).parent.parent


def test_reasoning_profile_omits_sampling_params() -> None:
    settings = ModelProfile(name="gpt-5.6", family="reasoning", reasoning_effort="high").settings()
    assert "temperature" not in settings
    assert "top_p" not in settings
    assert "seed" not in settings
    assert settings["reasoning"].effort == "high"
    ModelSettings(**settings)


def test_standard_profile_omits_reasoning() -> None:
    settings = ModelProfile(name="gpt-4.1", family="standard", temperature=0.3).settings()
    assert "reasoning" not in settings
    assert settings["temperature"] == 0.3
    ModelSettings(**settings)


def test_family_default_output_budgets() -> None:
    assert DEFAULT_MAX_OUTPUT_TOKENS["reasoning"] > DEFAULT_MAX_OUTPUT_TOKENS["standard"]
    profile = ModelProfile(name="gpt-5.6", family="reasoning", max_output_tokens=1234)
    assert profile.settings()["max_tokens"] == 1234


def test_family_inferred_from_name() -> None:
    assert ModelProfile(name="gpt-5.6").family == "reasoning"
    assert ModelProfile(name="o3").family == "reasoning"
    assert ModelProfile(name="gpt-4o").family == "standard"
    assert ModelProfile(name="gpt-4o", family="reasoning").family == "reasoning"


def test_cli_rejects_flags_for_wrong_model_kind() -> None:
    with pytest.raises(SystemExit):
        _profile_from_args(_parse_args(["run", "--model", "gpt-5.6", "--temperature", "0.2"]))
    with pytest.raises(SystemExit):
        _profile_from_args(_parse_args(["run", "--model", "gpt-4o", "--reasoning-effort", "high"]))


def test_cli_flags_override_config() -> None:
    config = RECIPE_DIR / "configs" / "model.toml"
    profile = _profile_from_args(
        _parse_args(["run", "--config", str(config), "--reasoning-effort", "high", "--max-output-tokens", "1234"])
    )
    assert profile.reasoning_effort == "high"
    assert profile.max_output_tokens == 1234
    profile = _profile_from_args(
        _parse_args(["run", "--config", str(config), "--model", "gpt-4.1", "--temperature", "0.7"])
    )
    assert profile.temperature == 0.7
    with pytest.raises(SystemExit):
        _profile_from_args(
            _parse_args(["run", "--config", str(config), "--model", "gpt-4.1", "--reasoning-effort", "high"])
        )


def test_invalid_effort_rejected() -> None:
    with pytest.raises(ValueError):
        ModelProfile(name="gpt-5.6", family="reasoning", reasoning_effort="ultra")


def test_example_configs_load() -> None:
    config = RECIPE_DIR / "configs" / "model.toml"
    default = ModelProfile.from_toml(config)
    assert default.name == "gpt-5.6"
    assert default.family == "reasoning" and default.reasoning_effort == "medium"
    standard = ModelProfile.from_toml(config, model="gpt-4.1")
    assert standard.family == "standard" and standard.temperature == 0.0
    with pytest.raises(ValueError):
        ModelProfile.from_toml(config, model="not-a-model")


def test_agent_wiring() -> None:
    from appworld_openai_agents_sdk.code_agent import run_code_agent_on_tasks  # noqa: F401

    assert MAX_STEPS == 50
    assert (PROMPTS_DIR / "react_code_agent" / "instructions.txt").is_file()


def test_vendored_files_self_contained() -> None:
    vendored_dir = RECIPE_DIR / "src" / "appworld_openai_agents_sdk" / "vendored"
    for path in vendored_dir.rglob("*.py"):
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                assert "appworld_agents" not in stripped, f"{path} still imports the un-packaged upstream tree"
