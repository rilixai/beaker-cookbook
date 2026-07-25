"""Local batch evaluation for the APEX-Agents agent.

Runs the :class:`~apex_agents.agent.agent.ApexReActAgent` over a set of
:class:`~apex_agents.data.dataset.ApexAgentsRecord` tasks and grades each
final answer with the rubric judge, aggregating to the benchmark metric
(``rubric_pass_rate``).

Tasks run concurrently (bounded by ``max_concurrency``). One task failing
never aborts the batch: an errored task counts as ``0`` (a real failure must
deflate, never inflate the metric); a task whose rubric has no scoreable
criteria is *unscoreable* and excluded from the aggregate entirely.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..agent.agent import ApexReActAgent
from ..data.dataset import ApexAgentsRecord
from .scoring import RUBRIC_FIELD, RubricJudge, score_rubric


logger = logging.getLogger(__name__)


__all__ = [
    "EvalReport",
    "evaluate_agent_on_records",
    "evaluate_record",
    "run_evaluation",
]


@dataclass
class EvalReport:
    """Aggregate of the agent's graded performance over a set of records."""

    num_cases: int
    rubric_pass_rate: float
    num_scored: int = 0
    num_errored: int = 0
    num_unscoreable: int = 0
    per_case: list[dict[str, Any]] = field(default_factory=list)


async def evaluate_record(
    *,
    record: ApexAgentsRecord,
    agent: ApexReActAgent,
    judge: RubricJudge,
) -> dict[str, Any]:
    """Run the agent on ``record`` and grade it, returning a per-case result.

    The returned dict always carries ``task_id`` + ``world_id`` and a ``kind``
    in ``{"scored", "error", "unscoreable"}``. The agent reports a failed run
    as an output carrying ``extra["error"]`` rather than raising, so that case
    is turned into an ``error`` row here (never graded). A ``scored`` result
    also carries ``rubric_pass_rate`` / ``n_criteria`` plus the agent's
    ``final_answer`` and loop telemetry.
    """
    output = await agent.forward(record=record)
    rubric_payload = [{"verifier_id": c.verifier_id, "criteria": c.criteria} for c in record.rubric]
    base = {"task_id": record.task_id, "world_id": record.world_id, "domain": record.domain}
    if not any(str(c.get("criteria") or "").strip() for c in rubric_payload):
        # No scoreable criteria: the task is not measurable, so it is kept
        # out of the aggregate rather than counted as a failure.
        return {**base, "kind": "unscoreable", "final_answer": output.final_answer}
    agent_error = str(output.extra.get("error") or "")
    if agent_error:
        return {**base, "kind": "error", "error": f"{output.status}: {agent_error}"}
    pass_rate = await asyncio.to_thread(
        score_rubric,
        rubric=rubric_payload,
        answer=output.final_answer,
        task_prompt=record.prompt,
        judge=judge,
    )
    return {
        **base,
        "kind": "scored",
        RUBRIC_FIELD: pass_rate,
        "n_criteria": len(rubric_payload),
        "status": output.status,
        "final_answer": output.final_answer,
        "total_steps": output.total_steps,
        "total_cost": output.total_cost,
        "wall_seconds": output.wall_seconds,
        "resum_count": output.resum_count,
    }


async def evaluate_agent_on_records(
    *,
    agent: ApexReActAgent,
    records: Sequence[ApexAgentsRecord],
    judge: RubricJudge,
    max_concurrency: int = 4,
) -> EvalReport:
    """Evaluate ``agent`` over ``records`` and aggregate ``rubric_pass_rate``."""
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def _run_one(record: ApexAgentsRecord) -> dict[str, Any]:
        async with semaphore:
            try:
                return await evaluate_record(record=record, agent=agent, judge=judge)
            except Exception as exc:  # noqa: BLE001 - one bad task must not abort the batch
                logger.exception("evaluate_record raised for task %s", record.task_id)
                return {
                    "task_id": record.task_id,
                    "world_id": record.world_id,
                    "domain": record.domain,
                    "kind": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }

    per_case = await asyncio.gather(*[_run_one(record) for record in records])

    rate_total = 0.0
    # Denominator = scored + errored. An errored task counts as 0 (a real
    # failure must deflate, never inflate). An unscoreable task (empty rubric)
    # is excluded entirely — it is not measurable.
    denominator = 0
    num_scored = 0
    num_errored = 0
    num_unscoreable = 0
    for result in per_case:
        kind = result.get("kind")
        if kind == "error":
            num_errored += 1
            denominator += 1  # denominator grows, the total stays 0
        elif kind == "unscoreable":
            num_unscoreable += 1
        else:
            num_scored += 1
            denominator += 1
            rate_total += float(result.get(RUBRIC_FIELD, 0.0))

    return EvalReport(
        num_cases=len(per_case),
        rubric_pass_rate=(rate_total / denominator) if denominator else 0.0,
        num_scored=num_scored,
        num_errored=num_errored,
        num_unscoreable=num_unscoreable,
        per_case=list(per_case),
    )


def run_evaluation(
    *,
    agent: ApexReActAgent,
    records: Sequence[ApexAgentsRecord],
    judge: RubricJudge,
    max_concurrency: int = 4,
) -> EvalReport:
    """Synchronous wrapper around :func:`evaluate_agent_on_records`."""
    return asyncio.run(
        evaluate_agent_on_records(
            agent=agent,
            records=records,
            judge=judge,
            max_concurrency=max_concurrency,
        )
    )
