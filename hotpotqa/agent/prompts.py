"""Default prompts for the PydanticAI agent benchmark.

An idiomatic two-tool agent: ``retrieve_k`` + ``summarize`` (with an
optional ``context`` arg). These are the source defaults the agent starts
with; the rilixai runner decides which of them are optimizer components.

The defaults below match the spirit of the paper's DSPy defaults: minimal
one-liners that don't pre-bake any tool-use strategy. Optimization
lift then measures what GEPA discovers about *agent* behavior, not
what handwritten prose already encodes.
"""

from __future__ import annotations


DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT = (
    "Given the question, use the available tools to retrieve evidence, then produce the structured answer."
)

DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT = (
    "Given the question and passages (optionally with prior context), produce a summary."
)
