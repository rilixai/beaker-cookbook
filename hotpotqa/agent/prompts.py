"""Seed prompts for the PydanticAI agent benchmark.

An idiomatic two-tool agent: ``retrieve_k`` + ``summarize`` (with an
optional ``context`` arg). Two optimizable components — the agent's
policy and the summarize tool's system prompt.

The seeds below match the spirit of the paper's DSPy defaults: minimal
one-liners that don't pre-bake any tool-use strategy. Optimization
lift then measures what GEPA discovers about *agent* behavior, not
what handwritten prose already encodes.
"""

from __future__ import annotations

from rilixai import OptimizationTargets, optimization_targets_from_prompts

from .types import POLICY_COMPONENT, SUMMARIZE_COMPONENT


# Re-export the canonical component names defined in ``types.py`` under the
# legacy ``*_PROMPT_COMPONENT`` spellings kept for external importers. The
# constants themselves live in ``types.py`` (single source of truth) so the
# two names can't drift out of sync.
POLICY_PROMPT_COMPONENT = POLICY_COMPONENT
SUMMARIZE_PROMPT_COMPONENT = SUMMARIZE_COMPONENT


DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT = (
    "Given the question, use the available tools to retrieve evidence, then produce the structured answer."
)

DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT = (
    "Given the question and passages (optionally with prior context), produce a summary."
)


def hotpotqa_pydantic_agent_seed_targets() -> OptimizationTargets:
    """Agent-mode seed :class:`OptimizationTargets` (2 components: policy + summarize)."""
    return optimization_targets_from_prompts(
        {
            POLICY_PROMPT_COMPONENT: DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT,
            SUMMARIZE_PROMPT_COMPONENT: DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT,
        }
    )


__all__ = [
    "DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT",
    "DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT",
    "POLICY_PROMPT_COMPONENT",
    "SUMMARIZE_PROMPT_COMPONENT",
    "hotpotqa_pydantic_agent_seed_targets",
]
