"""PydanticAI agent for the HotpotQA task.

An idiomatic two-tool agent: ``retrieve_k`` (BM25 over the case's
distractor paragraphs or the fullwiki bm25s index, depending on
``HotpotQAConfig.retrieval_mode``) + ``summarize`` (raw OpenAI
chat-completions call with the summarize prompt). The agent terminates
by populating a Pydantic ``HotpotQAOutput`` — PydanticAI's built-in
``final_result`` mechanism.

Two prompts steer it — ``policy_prompt`` + ``summarize_prompt`` — both
settable on the agent constructor (:mod:`hotpotqa.agent.prompts` holds
the defaults).
"""

from .agent import (
    HotpotQAOutput,
    HotpotQAPydanticAgent,
    SummarizeLLMCall,
)
from .prompts import (
    DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT,
    DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT,
    hotpotqa_default_prompts,
)
from .types import AgentToolCall, HotpotQAAgentOutput


__all__ = [
    "DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT",
    "DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT",
    "AgentToolCall",
    "HotpotQAAgentOutput",
    "HotpotQAOutput",
    "HotpotQAPydanticAgent",
    "SummarizeLLMCall",
    "hotpotqa_default_prompts",
]
