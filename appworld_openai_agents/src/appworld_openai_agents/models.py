"""Capability-aware OpenAI model layer.

Every model this recipe runs is described by a :class:`ModelProfile` whose
``family`` is the single switch that decides which sampling parameters are
attached to a request:

* ``reasoning`` (GPT-5 family, o-series): a ``reasoning={"effort": ...}``
  block is sent; ``temperature`` / ``top_p`` / ``seed`` are OMITTED (the API
  rejects them with a 400 for these models).
* ``standard`` (gpt-4.1, gpt-4o, ...): ``temperature`` / ``top_p`` / ``seed``
  are sent as usual; no ``reasoning`` block.

Unsupported parameters are gracefully omitted rather than sent-and-caught.
Models not in :data:`MODEL_PROFILES` fall back to a prefix heuristic
(:func:`infer_family`), so a brand-new snapshot id still routes correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from openai.types.shared import Reasoning


ModelFamily = Literal["reasoning", "standard"]

REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class ModelProfile:
    """Capability profile for one OpenAI model (or snapshot) id."""

    name: str
    family: ModelFamily


# Explicit registry. Extend freely; anything missing falls back to
# infer_family()'s prefix heuristic.
MODEL_PROFILES: dict[str, ModelProfile] = {
    profile.name: profile
    for profile in [
        # Reasoning models (GPT-5 family).
        ModelProfile("gpt-5.6", "reasoning"),
        ModelProfile("gpt-5.6-sol", "reasoning"),
        ModelProfile("gpt-5.6-terra", "reasoning"),
        ModelProfile("gpt-5.6-luna", "reasoning"),
        ModelProfile("gpt-5", "reasoning"),
        ModelProfile("gpt-5-mini", "reasoning"),
        ModelProfile("gpt-5-nano", "reasoning"),
        # Non-reasoning models.
        ModelProfile("gpt-4.1", "standard"),
        ModelProfile("gpt-4.1-mini", "standard"),
        ModelProfile("gpt-4.1-nano", "standard"),
        ModelProfile("gpt-4o", "standard"),
        ModelProfile("gpt-4o-mini", "standard"),
    ]
}


def infer_family(model_name: str) -> ModelFamily:
    """Prefix heuristic for models not in the registry."""
    base = model_name.lower()
    if base.startswith(("gpt-5", "o1", "o3", "o4")):
        return "reasoning"
    return "standard"


def resolve_profile(model_name: str, family: ModelFamily | None = None) -> ModelProfile:
    """Look up (or infer) the capability profile for ``model_name``.

    An explicit ``family`` (from a config file or CLI flag) always wins.
    """
    if family is not None:
        return ModelProfile(model_name, family)
    profile = MODEL_PROFILES.get(model_name)
    if profile is not None:
        return profile
    return ModelProfile(model_name, infer_family(model_name))


def build_model_settings(
    profile: ModelProfile,
    *,
    reasoning_effort: str = "medium",
    temperature: float = 0.0,
    top_p: float | None = None,
    seed: int | None = 100,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    """Assemble the kwargs for ``agents.ModelSettings`` for this model.

    Only parameters the model family supports are included; the rest are
    omitted (never sent-and-400'd). ``reasoning_effort`` is ignored for
    standard models and ``temperature``/``top_p``/``seed`` are ignored for
    reasoning models.
    """
    settings: dict[str, Any] = {"store": False}
    if max_output_tokens is not None:
        settings["max_tokens"] = max_output_tokens
    if profile.family == "reasoning":
        if reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"Invalid reasoning effort: {reasoning_effort!r}. Must be one of {REASONING_EFFORTS}.")
        settings["reasoning"] = Reasoning(effort=reasoning_effort)  # type: ignore[arg-type]
    else:
        settings["temperature"] = temperature
        if top_p is not None:
            settings["top_p"] = top_p
        if seed is not None:
            # ModelSettings has no first-class `seed`; it goes in extra_args.
            settings["extra_args"] = {"seed": seed}
    return settings
