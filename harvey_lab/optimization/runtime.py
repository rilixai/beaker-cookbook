"""Async rilixai ``run_case`` adapter for Harvey LAB — the optimizer surface.

Wraps :class:`~harvey_lab.agent.agent.HarveyLabAgent` as the async
:data:`rilixai.RunCase` the optimizer drives. Each call runs one
:class:`~rilixai.Case` end-to-end:

1. Unpack the :class:`HarveyLabRecord` off ``case.input`` and apply the
   :class:`~rilixai.OptimizationTargets` prompt bundle to the agent.
2. Run the Stirrup agent (workspace build, tool loop) → deliverables.
3. Grade every rubric criterion with the deliverable-scoped judge and
   aggregate to the all-pass result.
4. Return a :class:`~rilixai.CaseResult` whose ``output`` carries the
   ``all_pass`` + ``criterion_pass_rate`` fields the scorer reads back.

Decoupled from the task source + judge so tests inject a fixture-backed
workspace factory + scripted model + stub judge (zero network).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from rilixai import Case, CaseResult, OptimizationTargets, RunCase

from ..agent.agent import HarveyLabAgent, ModelFactory
from ..agent.types import HarveyLabAgentOutput
from ..agent.workspace import TaskSource
from ..config import HarveyLabConfig
from ..data.dataset import HarveyLabRecord
from .scoring import (
    ALL_PASS_FIELD,
    CRITERION_PASS_RATE_FIELD,
    CriterionJudge,
    build_criterion_judge,
    score_all_pass,
)


logger = logging.getLogger(__name__)


def build_harvey_lab_run_case(
    *,
    config: HarveyLabConfig | None = None,
    agent: HarveyLabAgent | None = None,
    task_source: TaskSource | None = None,
    model_factory: ModelFactory | None = None,
    judge: CriterionJudge | None = None,
) -> RunCase:
    """Build the async :data:`rilixai.RunCase` for the Harvey LAB agent.

    ``task_source`` is required when ``agent`` is None — it materializes each
    task's document workspace without depending on the network at import time.
    Tests pass a fixture-backed source + a scripted model factory. ``judge``
    defaults to the litellm-backed per-criterion judge; tests inject a stub.
    """
    cfg = config or HarveyLabConfig()

    resolved_agent: HarveyLabAgent
    if isinstance(agent, HarveyLabAgent):
        resolved_agent = agent
    elif agent is None:
        if task_source is None:
            raise ValueError(
                "build_harvey_lab_run_case requires either a pre-built agent or a "
                "task_source so the runtime can materialize per-case workspaces."
            )
        resolved_agent = HarveyLabAgent(config=cfg, task_source=task_source, model_factory=model_factory)
    else:
        raise TypeError(f"agent must be a HarveyLabAgent or None, got {type(agent).__name__}.")

    resolved_judge = (
        judge if judge is not None else build_criterion_judge(model=cfg.judge_model, timeout=cfg.llm_timeout)
    )

    async def _run_case(*, case: Case, targets: OptimizationTargets, runtime: Any = None) -> CaseResult:
        del runtime
        record = _record_from_case(case)
        # Pass the candidate's prompt components straight into ``forward`` rather
        # than mutating shared agent state first. The optimizer runs cases on
        # worker threads, so an ``apply_candidate``-then-``forward`` split on the
        # reused agent is racy: a concurrent case evaluating a *different*
        # candidate could swap the shared prompts between the two calls. Threading
        # the candidate through keeps each rollout isolated.
        output = await resolved_agent.forward(record=record, components=targets.to_dict())

        criteria_payload = [
            {"id": c.id, "title": c.title, "match_criteria": c.match_criteria, "deliverables": list(c.deliverables)}
            for c in record.criteria
        ]
        task_description = _task_description(record)
        has_scoreable = any(str(c.get("match_criteria") or "").strip() for c in criteria_payload)
        # A task with no scoreable criteria is unscoreable — signal it to the
        # scorer with ``None`` (rather than a real 0.0) so it is excluded from
        # the aggregates instead of deflating the all-pass metric.
        if not has_scoreable:
            return CaseResult(
                output={ALL_PASS_FIELD: None, CRITERION_PASS_RATE_FIELD: None, "final_answer": output.final_answer},
                run_metrics=_run_metrics(record=record, output=output, scored=None),
            )
        scored = await asyncio.to_thread(
            score_all_pass,
            criteria=criteria_payload,
            deliverables=output.deliverables,
            task_description=task_description,
            judge=resolved_judge,
            max_deliverable_chars=cfg.max_deliverable_chars,
        )
        return CaseResult(
            output={
                ALL_PASS_FIELD: scored["all_pass"],
                CRITERION_PASS_RATE_FIELD: scored["criterion_pass_rate"],
                "final_answer": output.final_answer,
            },
            run_metrics=_run_metrics(record=record, output=output, scored=scored),
        )

    return _run_case


def _task_description(record: HarveyLabRecord) -> str:
    title = record.title.strip()
    header = f"{title}\n\n" if title else ""
    return f"{header}{record.instructions}".strip()


def _run_metrics(
    *,
    record: HarveyLabRecord,
    output: HarveyLabAgentOutput,
    scored: dict[str, Any] | None,
) -> dict[str, Any]:
    """Translate the agent's output + grading into rilixai trajectory metadata."""
    return {
        "harvey_lab": {
            "task_id": record.task_id,
            "practice_area": record.practice_area,
            "work_type": record.work_type,
            "status": output.status,
            "num_criteria": len(record.criteria),
            "num_documents": len(record.documents),
            "deliverables_produced": sorted(output.deliverables),
            "total_turns": output.total_turns,
            "wall_seconds": output.wall_seconds,
            "answer_chars": len(output.final_answer),
            **({"n_passed": scored["n_passed"], "n_total": scored["n_total"]} if scored else {}),
            **({"all_pass": scored["all_pass"]} if scored else {}),
        },
    }


def _record_from_case(case: Case) -> HarveyLabRecord:
    record = case.input
    if not isinstance(record, HarveyLabRecord):
        raise TypeError(
            f"Harvey LAB run_case expected `case.input` to be a HarveyLabRecord, got {type(record).__name__}."
        )
    return record


__all__ = ["build_harvey_lab_run_case"]
