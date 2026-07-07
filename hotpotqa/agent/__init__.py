"""PydanticAI agent for the HotpotQA benchmark.

An idiomatic two-tool agent: ``retrieve_k`` (BM25 over local
distractor paragraphs or the fullwiki bm25s index, depending on
``HotpotQAConfig.retrieval_mode``) + ``summarize`` (raw OpenAI
chat-completions call with the optimizable summarize prompt). The
agent terminates by populating a Pydantic ``HotpotQAOutput`` —
PydanticAI's built-in ``final_result`` mechanism.

Two optimizable components — ``policy_prompt`` + ``summarize_prompt``
— get rewritten by rilixai's GEPA loop. The per-component feedback
strings the reflection LM reads live in :mod:`hotpotqa.optimization.feedback`
(they're GEPA-facing infrastructure, not agent internals the agent
itself reads).
"""

from .agent import (
    PYDANTIC_AGENT_POLICY_COMPONENT,
    PYDANTIC_AGENT_SUMMARIZE_COMPONENT,
    HotpotQAOutput,
    HotpotQAPydanticAgent,
    SummarizeLLMCall,
)
from .prompts import (
    DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT,
    DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT,
    hotpotqa_pydantic_agent_seed_targets,
)
from .types import AgentToolCall, HotpotQAAgentOutput


__all__ = [
    "AgentToolCall",
    "DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT",
    "DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT",
    "HotpotQAAgentOutput",
    "HotpotQAOutput",
    "HotpotQAPydanticAgent",
    "PYDANTIC_AGENT_POLICY_COMPONENT",
    "PYDANTIC_AGENT_SUMMARIZE_COMPONENT",
    "SummarizeLLMCall",
    "hotpotqa_pydantic_agent_seed_targets",
]
