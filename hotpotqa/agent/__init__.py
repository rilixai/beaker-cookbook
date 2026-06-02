"""PydanticAI agent for the HotpotQA benchmark.

An idiomatic two-tool agent: ``retrieve_k`` (BM25 over local
distractor paragraphs or the fullwiki bm25s index, depending on
``HotpotQAConfig.retrieval_mode``) + ``summarize`` (raw OpenAI
chat-completions call). The agent terminates by populating a Pydantic
``HotpotQAOutput`` — PydanticAI's built-in ``final_result`` mechanism.

The rilixai component mapping lives in :mod:`hotpotqa.rilixai_spec`; this
package stays focused on the agent itself.
"""

from .agent import (
    HotpotQAOutput,
    HotpotQAPydanticAgent,
    SummarizeLLMCall,
)
from .prompts import (
    DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT,
    DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT,
)
from .types import AgentToolCall, HotpotQAAgentOutput


__all__ = [
    "AgentToolCall",
    "DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT",
    "DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT",
    "HotpotQAAgentOutput",
    "HotpotQAOutput",
    "HotpotQAPydanticAgent",
    "SummarizeLLMCall",
]
