"""Capability-aware model configuration.

Each model belongs to a family — ``reasoning`` or ``standard`` — which decides
which sampling parameters are attached. The family is inferred from the model
name (GPT-5 / o-series → reasoning, everything else → standard), so sweeping
models is just a name change; an explicit ``family`` override remains available
for names the inference does not know about:

* ``reasoning`` models reject ``temperature`` / ``top_p`` / ``seed`` (the API
  400s), and take a ``reasoning={"effort": ...}`` setting instead.
* ``standard`` models take ``temperature`` / ``top_p`` as usual and reject the
  ``reasoning`` field.

Unsupported parameters are *omitted*, never sent-and-caught. The output of
:meth:`ModelProfile.to_model_config` is the ``model`` dict the vendored
AppWorld ``openai_agents`` runner expects (``type`` / ``name`` / ``settings``).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from openai.types.shared import Reasoning


ModelFamily = Literal["reasoning", "standard"]

# Model-name prefixes that identify OpenAI reasoning models.
_REASONING_NAME_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def infer_family(model_name: str) -> ModelFamily:
    """Infer the capability family from the model name."""
    return "reasoning" if model_name.startswith(_REASONING_NAME_PREFIXES) else "standard"


REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

# Sensible per-family defaults for the per-request output-token budget.
# Reasoning tokens count toward the output budget, so reasoning models get a
# much larger default to avoid truncated trajectories.
DEFAULT_MAX_OUTPUT_TOKENS: dict[ModelFamily, int] = {
    "reasoning": 65536,
    "standard": 16384,
}


@dataclass
class ModelProfile:
    """One model entry: its identity plus the capability profile that decides
    which sampling parameters may be attached."""

    name: str
    family: ModelFamily | None = None  # inferred from name if not set
    # `responses` is the OpenAI Agents SDK default for native OpenAI models and
    # serves both reasoning and non-reasoning models; it is what upstream's
    # `type: openai` routes to (a plain model-name string resolved by the SDK,
    # NOT LitellmModel), so reasoning settings are honored.
    api_type: Literal["responses", "chat_completions"] = "responses"
    reasoning_effort: str = "medium"  # reasoning family only
    temperature: float = 0.0  # standard family only
    top_p: float | None = None  # standard family only
    max_output_tokens: int | None = None  # per model request; family default if None

    def __post_init__(self) -> None:
        if self.family is None:
            self.family = infer_family(self.name)
        if self.family not in ("reasoning", "standard"):
            raise ValueError(f"Unknown model family: {self.family!r}")
        if self.family == "reasoning" and self.reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(
                f"Invalid reasoning effort {self.reasoning_effort!r}; expected one of {REASONING_EFFORTS}."
            )

    def settings(self, for_agent: bool = True) -> dict[str, Any]:
        """The ``settings`` dict consumed by the vendored runner (it feeds
        ``ModelSettings(**settings)`` after popping the routing keys)."""
        family: ModelFamily = self.family or infer_family(self.name)
        max_output_tokens = self.max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS[family]
        settings: dict[str, Any] = {"store": False}
        if for_agent:
            # Routing keys popped by the vendored runner before it builds
            # ModelSettings. The predictor's LanguageModel builds ModelSettings
            # directly, so it must not see them.
            settings["api_type"] = self.api_type
        if not for_agent and family == "reasoning":
            # The predictor rides Chat Completions (upstream's LanguageModel
            # uses OpenAIChatCompletionsModel), where reasoning models reject
            # `max_tokens` and require `max_completion_tokens` instead.
            settings["extra_args"] = {"max_completion_tokens": max_output_tokens}
        else:
            settings["max_tokens"] = max_output_tokens
        if family == "reasoning":
            settings["reasoning"] = Reasoning(effort=self.reasoning_effort)  # type: ignore[arg-type]
        else:
            settings["temperature"] = self.temperature
            if self.top_p is not None:
                settings["top_p"] = self.top_p
        if for_agent:
            # Upstream reference config (openai_agents_mcp_agent): tool_choice
            # auto + parallel tool calls, for all models.
            settings["tool_choice"] = "auto"
            settings["parallel_tool_calls"] = True
        return settings

    def to_model_config(self, for_agent: bool = True) -> dict[str, Any]:
        # `type: openai` + a plain name string routes to the SDK's native
        # OpenAI model classes (Responses by default), not LitellmModel.
        return {
            "type": "openai",
            "name": self.name,
            "settings": self.settings(for_agent=for_agent),
            "extras": {},
        }

    @classmethod
    def from_toml(cls, path: Path) -> ModelProfile:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        model = data.get("model")
        if not isinstance(model, dict):
            raise ValueError(f"{path}: expected a [model] table.")
        return cls(**model)
