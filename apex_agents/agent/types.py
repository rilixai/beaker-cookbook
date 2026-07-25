"""Shared dataclasses for the APEX-Agents agent benchmark.

Framework-neutral so downstream code doesn't need to import litellm
just to type-check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentToolCall:
    """One step of the ReAct toolbelt loop, normalized for reporting.

    * ``role`` — the message role (``"system"``, ``"user"``,
      ``"assistant"``, or ``"tool"``).
    * ``content`` — the message body (the model's reasoning text, a
      tool result, etc.).
    * ``tool_name`` — for assistant messages that issued a tool call,
      the called tool's name; ``None`` otherwise.
    * ``tool_args`` — the parsed JSON arguments of that tool call.
    * ``output`` — for tool messages, the tool's textual result.
    """

    step_index: int
    role: str
    content: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    output: str | None = None


@dataclass
class ApexAgentsAgentOutput:
    """Per-case result returned by the APEX-Agents ReAct agent.

    Empty / missing data is represented with empty strings / lists so
    downstream code never has to ``None``-check. The rubric score is
    NOT computed here — :mod:`apex_agents.evaluation` runs the LLM judge
    after the agent terminates.
    """

    final_answer: str
    status: str
    messages: list[AgentToolCall] = field(default_factory=list)
    total_steps: int = 0
    total_cost: float = 0.0
    wall_seconds: float = 0.0
    resum_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


__all__ = ["AgentToolCall", "ApexAgentsAgentOutput"]
