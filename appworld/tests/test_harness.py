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
    settings = ModelProfile(name="gpt-5.6-sol", family="reasoning", reasoning_effort="high").settings()
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
    profile = ModelProfile(name="gpt-5.6-sol", family="reasoning", max_output_tokens=1234)
    assert profile.settings()["max_tokens"] == 1234


def test_family_inferred_from_name() -> None:
    assert ModelProfile(name="gpt-5.6-sol").family == "reasoning"
    assert ModelProfile(name="gpt-6-luna").family == "reasoning"
    assert ModelProfile(name="o3").family == "reasoning"
    assert ModelProfile(name="gpt-4o").family == "standard"
    assert ModelProfile(name="gpt-4o", family="reasoning").family == "reasoning"


def test_cli_rejects_flags_for_wrong_model_kind() -> None:
    with pytest.raises(SystemExit):
        _profile_from_args(_parse_args(["run", "--model", "gpt-5.6-sol", "--temperature", "0.2"]))
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
        ModelProfile(name="gpt-5.6-sol", family="reasoning", reasoning_effort="ultra")


def test_example_configs_load() -> None:
    config = RECIPE_DIR / "configs" / "model.toml"
    default = ModelProfile.from_toml(config)
    assert default.name == "gpt-6-luna"
    assert default.family == "reasoning" and default.reasoning_effort == "low"
    cli_default = _profile_from_args(_parse_args(["run"]))
    assert cli_default.name == default.name
    assert cli_default.settings()["reasoning"].effort == "low"
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


def test_endpoint_profile_leaves_sampling_to_provider() -> None:
    profile = ModelProfile(
        name="router-model",
        model="vendor-model",
        base_url="https://router.example/v1",
        api_key_env="ROUTER_KEY",
        extra_body={"reasoning_effort": "high"},
    )
    settings = profile.settings()
    assert settings == {"tool_choice": "auto", "extra_body": {"reasoning_effort": "high"}}
    assert profile.api_type == "chat_completions"
    assert profile.api_model == "vendor-model"
    ModelSettings(**settings)
    with pytest.raises(ValueError):
        ModelProfile(name="no-key", base_url="https://router.example/v1")


def test_endpoint_run_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from agents import OpenAIChatCompletionsModel
    from openai import AsyncOpenAI

    from appworld_openai_agents_sdk.runner import endpoint_client, endpoint_run_config

    profile = ModelProfile(
        name="router-model", model="vendor-model", base_url="https://router.example/v1", api_key_env="ROUTER_KEY"
    )
    monkeypatch.delenv("ROUTER_KEY", raising=False)
    with pytest.raises(RuntimeError):
        endpoint_client(profile)
    monkeypatch.setenv("ROUTER_KEY", "test-key")
    client = endpoint_client(profile)
    assert str(client.base_url).rstrip("/") == "https://router.example/v1"
    model = endpoint_run_config(profile, AsyncOpenAI(base_url="https://router.example/v1", api_key="test")).model
    assert isinstance(model, OpenAIChatCompletionsModel) and model.model == "vendor-model"


def test_endpoint_example_configs_load() -> None:
    config = RECIPE_DIR / "configs" / "model.toml"
    youcom = ModelProfile.from_toml(config, model="youcom-deepseek-v4.1-flash")
    assert youcom.api_model == "deepseek-v4.1-flash" and youcom.api_key_env == "YDC_API_KEY"
    assert youcom.settings()["extra_body"] == {"reasoning_effort": "high"}
    openrouter = ModelProfile.from_toml(config, model="openrouter-glm-5.3-flash")
    assert openrouter.api_model == "z-ai/glm-5.3-flash"
    assert openrouter.settings()["extra_body"] == {"reasoning": {"effort": "high"}}


def _hosted_module(name: str):
    import importlib
    import sys

    sys.path.insert(0, str(RECIPE_DIR / ".beaker"))
    return importlib.import_module(name)


def test_hosted_model_setting_is_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    setup = _hosted_module("appworld_setup")
    no_model = SimpleNamespace(model=None)
    monkeypatch.delenv("APPWORLD_MODEL", raising=False)
    setup.check_model_setting(no_model)
    monkeypatch.setenv("APPWORLD_MODEL", "no-such-table")
    with pytest.raises(ValueError):
        setup.check_model_setting(no_model)
    setup.check_model_setting(SimpleNamespace(model={"provider": "openrouter"}))
    monkeypatch.setenv("APPWORLD_MODEL", "youcom-glm-5.3-flash")
    monkeypatch.delenv("YDC_API_KEY", raising=False)
    with pytest.raises(ValueError):
        setup.check_model_setting(no_model)
    monkeypatch.setenv("YDC_API_KEY", "test-key")
    setup.check_model_setting(no_model)


def test_hosted_worker_calls_the_selected_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    from types import SimpleNamespace

    from agents import OpenAIChatCompletionsModel

    worker = _hosted_module("appworld_case_worker")
    monkeypatch.setenv("APPWORLD_MODEL", "youcom-glm-5.3-flash")
    monkeypatch.setenv("YDC_API_KEY", "test-key")

    async def selected() -> None:
        async with worker._model_config(SimpleNamespace(model=None)) as (profile, run_config):
            assert profile.api_model == "glm-5.3-flash"
            assert isinstance(run_config.model, OpenAIChatCompletionsModel)
            assert run_config.model.model == "glm-5.3-flash"
            assert not run_config.tracing_disabled

    asyncio.run(selected())
