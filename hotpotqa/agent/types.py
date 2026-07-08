"""Shared dataclasses for the HotpotQA PydanticAI agent.

Framework-neutral so the rilixai trajectory translation and the
feedback functions don't need to import ``pydantic_ai`` just to
type-check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..data.dataset import HotpotQAParagraph


# Component names the agent attaches to the rilixai OptimizationTargets.
# Defined in this framework-neutral module so the agent runtime
# (``agent.py``, which imports PydanticAI) and the GEPA-facing
# feedback functions (``optimization/feedback.py``) can both import
# them without dragging in either layer's heavy deps. Single source
# of truth — no aliases — so they can't drift.
POLICY_COMPONENT = "policy_prompt"
SUMMARIZE_COMPONENT = "summarize_prompt"


@dataclass
class AgentToolCall:
    """One step of a HotpotQA agent's tool-use loop, framework-agnostic."""

    step_index: int
    tool_name: str
    tool_args: dict[str, Any]
    observation: str
    thought: str
    # Gold supporting titles not yet retrieved *before* this step ran and
    # *after* it completed. Populated for retrieval steps; copy-through for
    # non-retrieval steps. Drives the paper-style per-step textual feedback.
    gold_titles_remaining_before: list[str] = field(default_factory=list)
    gold_titles_remaining_after: list[str] = field(default_factory=list)


@dataclass
class HotpotQAAgentOutput:
    """Per-case result returned by the HotpotQA PydanticAI agent.

    ``tool_calls`` is the authoritative record of what the agent did this
    pass — including any summarize calls (look for entries with
    ``tool_name == "summarize"``, with the produced summary stored in
    ``observation``). The agent decides how many times to call summarize,
    so there's no top-level ``summary_1`` / ``summary_2`` slot — that
    would imply a workflow-shaped fixed-hop pattern this agent doesn't
    have.
    """

    answer: str
    retrieved_paragraphs: list[HotpotQAParagraph]
    tool_calls: list[AgentToolCall]
