"""Framework-neutral result types for the Harvey LAB agent."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HarveyLabAgentOutput:
    """Per-case result returned by the Harvey LAB Stirrup agent.

    ``deliverables`` maps each ``output/`` filename the agent produced to its
    text content — this is what the rubric judge grades (criteria are scoped to
    named deliverables). The rubric score is NOT computed here; the runtime
    runs the per-criterion judge after the agent terminates.
    """

    final_answer: str
    status: str
    deliverables: dict[str, str] = field(default_factory=dict)
    total_turns: int = 0
    wall_seconds: float = 0.0


__all__ = ["HarveyLabAgentOutput"]
