"""Run the Harvey LAB agent over a set of tasks and grade each one.

Grades agent outputs over a set of
:class:`~harvey_lab.data.dataset.HarveyLabRecord` tasks with the batched rubric
judge, aggregating to the LAB metrics (``all_pass_rate`` +
``criterion_pass_rate``). Outputs may come directly from a
:class:`~harvey_lab.agent.agent.HarveyLabAgent` or from persisted run artifacts.

Cases run concurrently (bounded by ``max_concurrency``). One case failing
never aborts the batch: an errored case counts as ``0`` (a real failure must
deflate, never inflate the metrics); a case whose rubric has no scoreable
criteria is *unscoreable* and excluded from the aggregates entirely.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..agent.agent import HarveyLabAgent, HarveyLabAgentOutput
from ..data.dataset import HarveyLabRecord
from .scoring import (
    ALL_PASS_FIELD,
    CRITERION_PASS_RATE_FIELD,
    DEFAULT_JUDGE_BATCH_SIZE,
    BatchJudge,
    JudgeCallError,
    score_rubric,
)


logger = logging.getLogger(__name__)


@dataclass
class EvalReport:
    """Aggregate of the agent's graded performance over a set of records."""

    num_cases: int
    all_pass_rate: float
    criterion_pass_rate: float
    num_scored: int = 0
    num_errored: int = 0
    num_unscoreable: int = 0
    per_case: list[dict[str, Any]] = field(default_factory=list)


def _task_description(record: HarveyLabRecord) -> str:
    title = record.title.strip()
    header = f"{title}\n\n" if title else ""
    return f"{header}{record.instructions}".strip()


async def evaluate_output(
    *,
    record: HarveyLabRecord,
    output: HarveyLabAgentOutput,
    judge: BatchJudge,
    batch_size: int = DEFAULT_JUDGE_BATCH_SIZE,
) -> dict[str, Any]:
    """Grade one already-produced agent output against its task rubric."""
    criteria_payload = [
        {"id": c.id, "title": c.title, "match_criteria": c.match_criteria, "deliverables": list(c.deliverables)}
        for c in record.criteria
    ]
    has_scoreable = any(str(c.get("match_criteria") or "").strip() for c in criteria_payload)
    base = {"task_id": record.task_id, "practice_area": record.practice_area}
    if not has_scoreable:
        logger.info("Skipping unscoreable task %s: no scoreable rubric criteria.", record.task_id)
        return {**base, "kind": "unscoreable", "final_answer": output.final_answer}
    total_criteria = sum(1 for criterion in criteria_payload if str(criterion.get("match_criteria") or "").strip())
    logger.info(
        "Grading task %s (%d criteria, %d produced deliverables)...",
        record.task_id,
        total_criteria,
        len(output.deliverables),
    )

    def _log_judge_batch(start: int, end: int, total: int) -> None:
        logger.info(
            "Grading task %s: submitting criteria %d-%d of %d to the judge...", record.task_id, start, end, total
        )

    scored = await asyncio.to_thread(
        score_rubric,
        criteria=criteria_payload,
        deliverables=output.deliverables,
        task_description=_task_description(record),
        judge=judge,
        batch_size=batch_size,
        judge_batch_callback=_log_judge_batch,
    )
    logger.info(
        "Finished grading task %s: %d/%d criteria passed.",
        record.task_id,
        scored["passed"],
        scored["total_criteria"],
    )
    return {
        **base,
        "kind": "scored",
        ALL_PASS_FIELD: scored[ALL_PASS_FIELD],
        CRITERION_PASS_RATE_FIELD: scored[CRITERION_PASS_RATE_FIELD],
        "passed": scored["passed"],
        "total_criteria": scored["total_criteria"],
        "deliverables_produced": sorted(output.deliverables),
        "deliverables_missing": sorted(output.missing_deliverables),
        "finished": output.finished,
        "abandoned": output.abandoned,
        "max_turns_reached": output.max_turns_reached,
        "total_turns": output.total_turns,
        "final_answer": output.final_answer,
    }


async def evaluate_record(
    *,
    record: HarveyLabRecord,
    agent: HarveyLabAgent,
    judge: BatchJudge,
    batch_size: int = DEFAULT_JUDGE_BATCH_SIZE,
) -> dict[str, Any]:
    """Run the agent on ``record`` and grade its output."""
    output = await agent.forward(record=record)
    return await evaluate_output(
        record=record,
        output=output,
        judge=judge,
        batch_size=batch_size,
    )


def _aggregate_results(per_case: Sequence[dict[str, Any]]) -> EvalReport:
    all_pass_total = 0.0
    rate_total = 0.0
    denominator = 0
    num_scored = 0
    num_errored = 0
    num_unscoreable = 0
    for result in per_case:
        kind = result.get("kind")
        if kind == "error":
            num_errored += 1
            denominator += 1
        elif kind == "unscoreable":
            num_unscoreable += 1
        else:
            num_scored += 1
            denominator += 1
            all_pass_total += float(result.get(ALL_PASS_FIELD, 0.0))
            rate_total += float(result.get(CRITERION_PASS_RATE_FIELD, 0.0))
    return EvalReport(
        num_cases=len(per_case),
        all_pass_rate=(all_pass_total / denominator) if denominator else 0.0,
        criterion_pass_rate=(rate_total / denominator) if denominator else 0.0,
        num_scored=num_scored,
        num_errored=num_errored,
        num_unscoreable=num_unscoreable,
        per_case=list(per_case),
    )


async def evaluate_outputs_on_records(
    *,
    records: Sequence[HarveyLabRecord],
    outputs: Mapping[str, HarveyLabAgentOutput],
    errors: Mapping[str, str],
    judge: BatchJudge,
    batch_size: int = DEFAULT_JUDGE_BATCH_SIZE,
    max_concurrency: int = 4,
) -> EvalReport:
    """Grade persisted outputs without invoking an agent."""
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def _grade_one(record: HarveyLabRecord) -> dict[str, Any]:
        if record.task_id in errors:
            return {
                "task_id": record.task_id,
                "practice_area": record.practice_area,
                "kind": "error",
                "error": errors[record.task_id],
            }
        output = outputs.get(record.task_id)
        if output is None:
            return {
                "task_id": record.task_id,
                "practice_area": record.practice_area,
                "kind": "error",
                "error": "No persisted output is available.",
            }
        async with semaphore:
            try:
                return await evaluate_output(
                    record=record,
                    output=output,
                    judge=judge,
                    batch_size=batch_size,
                )
            except JudgeCallError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad case must not abort the batch
                logger.exception("evaluate_output raised for task %s", record.task_id)
                return {
                    "task_id": record.task_id,
                    "practice_area": record.practice_area,
                    "kind": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }

    per_case = await asyncio.gather(*[_grade_one(record) for record in records])
    return _aggregate_results(per_case)


async def evaluate_agent_on_records(
    *,
    agent: HarveyLabAgent,
    records: Sequence[HarveyLabRecord],
    judge: BatchJudge,
    batch_size: int = DEFAULT_JUDGE_BATCH_SIZE,
    max_concurrency: int = 4,
) -> EvalReport:
    """Evaluate ``agent`` over ``records`` and aggregate the LAB metrics."""
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def _run_one(record: HarveyLabRecord) -> dict[str, Any]:
        async with semaphore:
            try:
                return await evaluate_record(
                    record=record,
                    agent=agent,
                    judge=judge,
                    batch_size=batch_size,
                )
            except JudgeCallError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad case must not abort the batch
                logger.exception("evaluate_record raised for task %s", record.task_id)
                return {
                    "task_id": record.task_id,
                    "practice_area": record.practice_area,
                    "kind": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }

    per_case = await asyncio.gather(*[_run_one(record) for record in records])
    return _aggregate_results(per_case)


__all__ = [
    "EvalReport",
    "evaluate_agent_on_records",
    "evaluate_output",
    "evaluate_outputs_on_records",
    "evaluate_record",
]
