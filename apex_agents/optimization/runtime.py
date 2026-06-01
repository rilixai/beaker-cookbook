"""Trajectory metadata builder for the APEX-Agents agent.

Translates the agent's output + the precomputed ``rubric_pass_rate`` into the
``run_metrics`` dict the optimizer's adapter reads. The runtime closure +
result-wrapper that used to live here are gone:
:class:`~apex_agents.rilixai_spec.ApexAgentsRunner` (a
``rilixai.adapters.BaseSampleRunner`` subclass) now owns the run loop (agent
forward + judge) and calls :func:`build_apex_agents_run_metrics` from its
``_package_result``.
"""

from __future__ import annotations

import logging
from typing import Any

from ..agent.types import ApexAgentsAgentOutput
from ..config import ApexAgentsConfig
from ..data.dataset import ApexAgentsRecord


logger = logging.getLogger(__name__)


def build_apex_agents_run_metrics(
    *,
    record: ApexAgentsRecord,
    output: ApexAgentsAgentOutput,
    config: ApexAgentsConfig,
    rubric_pass_rate: float,
) -> dict[str, Any]:
    """Translate the agent's output into rilixai trajectory metadata.

    Per-component feedback is no longer embedded here — it flows through
    ``@spec(feedback=ApexAgentsFeedback)`` (the runner merges it into
    ``trace_evidence.per_component_feedback`` in ``_package_result``). This
    builder owns only the domain-specific trace evidence.
    """
    policy_reasoning = [
        _truncate(m.content, config.max_preview_chars) for m in output.messages if m.role == "assistant"
    ][:5]
    tools_called = [m.tool_name for m in output.messages if m.role == "assistant" and m.tool_name]

    tool_counts: dict[str, int] = {}
    for name in tools_called:
        tool_counts[name] = tool_counts.get(name, 0) + 1

    tool_calls_detail: list[dict[str, Any]] = []
    for step in output.messages:
        tool_calls_detail.append(
            {
                "step_index": step.step_index,
                "role": step.role,
                "tool_name": step.tool_name,
                "tool_args": step.tool_args,
                "output_preview": (_truncate(step.output or "", config.max_preview_chars) if step.output else None),
                "content_preview": _truncate(step.content, config.max_preview_chars),
            }
        )

    return {
        "tool_counts": tool_counts,
        "tool_calls_detail": tool_calls_detail,
        "trace_evidence": {
            "policy_reasoning": policy_reasoning,
            "tools_called": tools_called[:25],
        },
        "apex_agents": {
            "task_id": record.task_id,
            "world_id": record.world_id,
            "domain": record.domain,
            "status": output.status,
            "rubric_pass_rate": rubric_pass_rate,
            "num_rubric_criteria": len(record.rubric),
            "total_steps": output.total_steps,
            "total_cost": output.total_cost,
            "wall_seconds": output.wall_seconds,
            "resum_count": output.resum_count,
            "answer_chars": len(output.final_answer),
        },
    }


# ─── Shared helpers ──────────────────────────────────────────────────


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"
