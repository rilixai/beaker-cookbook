"""Async rilixai ``run_case`` adapter — the optimizer-facing surface.

Wraps :class:`~apex_agents.agent.agent.ApexReActAgent` as the async
:data:`rilixai.RunCase` the optimizer drives. Each call runs one
:class:`~rilixai.Case` end-to-end:

1. Unpack the :class:`ApexAgentsRecord` off ``case.input`` and apply the
   :class:`~rilixai.OptimizationTargets` prompt bundle to the agent.
2. Run the agent (world build, ReAct loop, ReSum — all in the agent).
3. Run the LLM judge over the rubric to compute ``rubric_pass_rate``.
4. Translate the agent's output + per-component feedback into a
   :class:`~rilixai.CaseResult` whose ``output`` carries the scored
   ``rubric_pass_rate`` field the :class:`ApexAgentsScorer` reads back.

Decoupled from the world factory + judge so tests inject a
:class:`FakeWorld` factory and a stub judge while production passes
the real HF builder + litellm judge.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from rilixai import Case, CaseResult, OptimizationTargets, RunCase

from ..agent.types import ApexAgentsAgentOutput
from ..config import ApexAgentsConfig
from ..data.dataset import ApexAgentsRecord
from .feedback import build_apex_per_component_feedback
from .metrics import RUBRIC_FIELD, build_rubric_judge, score_rubric


logger = logging.getLogger(__name__)


def build_apex_agents_run_case(
    *,
    config: ApexAgentsConfig | None = None,
    agent: Any | None = None,
    world_factory: Callable[[Any], Any] | None = None,
    model_factory: Callable[[str, float], Any] | None = None,
    judge: Callable[[str, str, str], bool] | None = None,
) -> RunCase:
    """Build the async :data:`rilixai.RunCase` for the APEX-Agents agent.

    ``world_factory`` is required when ``agent`` is None — it's how the
    run_case builds per-case worlds without depending on HuggingFace at
    import time. Tests pass a closure yielding :class:`FakeWorld`.
    ``judge`` defaults to the litellm-backed rubric judge; tests inject
    a stub so zero network fires.
    """
    cfg = config or ApexAgentsConfig()

    # Defer the import so callers that just need the runtime symbol
    # (e.g. spec construction at module-import time) don't pay the
    # cost until the runtime actually fires.
    from ..agent.agent import ApexReActAgent
    from ..agent.prompts import load_apex_agents_seed_prompts

    resolved_agent: ApexReActAgent
    if isinstance(agent, ApexReActAgent):
        resolved_agent = agent
    elif agent is None:
        if world_factory is None:
            raise ValueError(
                "build_apex_agents_run_case requires either a pre-built agent or a "
                "world_factory so the runtime can construct per-case worlds."
            )
        default_sys, default_task, default_resum = load_apex_agents_seed_prompts()
        resolved_agent = ApexReActAgent(
            model_name=cfg.task_model,
            model_temperature=cfg.task_temperature,
            max_steps=cfg.max_steps,
            cost_limit=cfg.cost_limit,
            max_toolbelt_size=cfg.max_toolbelt_size,
            max_context_tokens=cfg.max_context_tokens,
            default_system_prompt=default_sys,
            default_task_template=default_task,
            default_resum_summary_prompt=default_resum,
            world_factory=world_factory,
            model_factory=model_factory,
            llm_timeout=cfg.llm_timeout,
        )
    else:
        raise TypeError(f"agent must be an ApexReActAgent or None, got {type(agent).__name__}.")

    resolved_judge = judge if judge is not None else build_rubric_judge(model=cfg.judge_model, timeout=cfg.llm_timeout)

    async def _run_case(*, case: Case, targets: OptimizationTargets, runtime: Any = None) -> CaseResult:
        del runtime
        record = _record_from_case(case)
        resolved_agent.apply_candidate(targets.to_dict())
        output = await resolved_agent.forward(record=record)
        rubric_payload = [{"verifier_id": c.verifier_id, "criteria": c.criteria} for c in record.rubric]
        # A rubric with no non-blank criteria is unscoreable. Signal that to
        # the scorer with ``None`` (rather than a real ``0.0``) so it can omit
        # the field and keep the case out of the aggregate ``rubric_pass_rate``
        # — matching the pre-migration metrics layer, which excluded
        # rubric-less cases so they couldn't deflate the benchmark score.
        has_scoreable_criteria = any(str(c.get("criteria") or "").strip() for c in rubric_payload)
        rubric_pass_rate = await asyncio.to_thread(
            score_rubric,
            rubric=rubric_payload,
            answer=output.final_answer,
            task_prompt=record.prompt,
            judge=resolved_judge,
        )
        run_metrics = build_apex_agents_run_metrics(
            record=record,
            output=output,
            config=cfg,
            rubric_pass_rate=rubric_pass_rate,
        )
        return CaseResult(
            output={
                RUBRIC_FIELD: rubric_pass_rate if has_scoreable_criteria else None,
                "final_answer": output.final_answer,
            },
            run_metrics=run_metrics,
        )

    return _run_case


def build_apex_agents_run_metrics(
    *,
    record: ApexAgentsRecord,
    output: ApexAgentsAgentOutput,
    config: ApexAgentsConfig,
    rubric_pass_rate: float,
) -> dict[str, Any]:
    """Translate the agent's output into rilixai trajectory metadata."""
    per_component_feedback = build_apex_per_component_feedback(record=record, output=output)

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
            "per_component_feedback": per_component_feedback,
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


def _record_from_case(case: Case) -> ApexAgentsRecord:
    record = case.input
    if not isinstance(record, ApexAgentsRecord):
        raise TypeError(
            f"APEX-Agents run_case expected `case.input` to be an ApexAgentsRecord, got {type(record).__name__}."
        )
    return record


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"
