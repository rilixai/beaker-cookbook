"""The agent's two prompts.

An idiomatic two-tool agent: ``retrieve_k`` + ``summarize`` (with an
optional ``context`` arg). Two prompts steer it — the agent's tool-use
policy and the summarize tool's system prompt.

Both are deliberately minimal one-liners that pre-bake no tool-use
strategy, so the measured behavior is the agent's, not handwritten
prose's. Pass your own to :class:`~hotpotqa.agent.agent.HotpotQAPydanticAgent`
to try alternatives.
"""

from __future__ import annotations


DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT = (
    "Given the question, use the available tools to retrieve evidence, then produce the structured answer."
)

DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT = (
    "Given the question and passages (optionally with prior context), produce a summary."
)


def hotpotqa_default_prompts() -> dict[str, str]:
    """The default ``{policy_prompt, summarize_prompt}`` pair."""
    return {
        "policy_prompt": DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT,
        "summarize_prompt": DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT,
    }


__all__ = [
    "DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT",
    "DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT",
    "hotpotqa_default_prompts",
]
