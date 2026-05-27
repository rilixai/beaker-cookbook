"""PydanticAI agent variant of the HotpotQA benchmark.

The same multi-hop task as the workflow, expressed as an idiomatic
PydanticAI tool-using agent: two tools (``retrieve_k`` + ``summarize``
with an optional ``context`` arg), a structured Pydantic answer
terminator, and two optimizable components (``policy_prompt`` +
``summarize_prompt``). Per-component feedback (in :mod:`.feedback`) is
independent of the workflow's per-module feedback because the agent's
tool sequence is variable.
"""

from .agent import (
    PYDANTIC_AGENT_POLICY_COMPONENT,
    PYDANTIC_AGENT_SUMMARIZE_COMPONENT,
    HotpotQAOutput,
    HotpotQAPydanticAgent,
    SummarizeLLMCall,
)
from .feedback import (
    AGENT_POLICY_COMPONENT,
    AGENT_SUMMARIZE_COMPONENT,
    build_agent_per_component_feedback,
)
from .prompts import (
    DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT,
    DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT,
    hotpotqa_pydantic_agent_seed_candidate,
)
from .runtime import build_agent_run_metrics, build_pydantic_agent_runtime
from .types import AgentToolCall, HotpotQAAgentOutput


__all__ = [
    "AGENT_POLICY_COMPONENT",
    "AGENT_SUMMARIZE_COMPONENT",
    "AgentToolCall",
    "DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT",
    "DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT",
    "HotpotQAAgentOutput",
    "HotpotQAOutput",
    "HotpotQAPydanticAgent",
    "PYDANTIC_AGENT_POLICY_COMPONENT",
    "PYDANTIC_AGENT_SUMMARIZE_COMPONENT",
    "SummarizeLLMCall",
    "build_agent_per_component_feedback",
    "build_agent_run_metrics",
    "build_pydantic_agent_runtime",
    "hotpotqa_pydantic_agent_seed_candidate",
]
