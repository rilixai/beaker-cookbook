"""Local batch evaluation for the Harvey LAB agent.

Runs the :class:`~harvey_lab.agent.agent.HarveyLabAgent` over a set of
:class:`~harvey_lab.data.dataset.HarveyLabRecord` tasks and grades each with
the deliverable-scoped rubric judge, aggregating to the LAB metrics
(``all_pass`` + ``criterion_pass_rate``).

Cases run concurrently (bounded by ``max_concurrency``). One case failing
never aborts the batch: an errored case counts as ``0`` (a real failure must
deflate, never inflate the metrics); a case whose rubric has no scoreable
criteria is *unscoreable* and excluded from the aggregates entirely.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..agent.agent import HarveyLabAgent
from ..config import HarveyLabConfig
from ..data.dataset import HarveyLabRecord
from .scoring import (
    ALL_PASS_FIELD,
    CRITERION_PASS_RATE_FIELD,
    DEFAULT_MAX_DELIVERABLE_CHARS,
    CriterionJudge,
    score_all_pass,
)


logger = logging.getLogger(__name__)


@dataclass
class EvalReport:
    """Aggregate of the agent's graded performance over a set of records."""

    num_cases: int
    all_pass: float
    criterion_pass_rate: float
    num_scored: int = 0
    num_errored: int = 0
    num_unscoreable: int = 0
    per_case: list[dict[str, Any]] = field(default_factory=list)


def _task_description(record: HarveyLabRecord) -> str:
    title = record.title.strip()
    header = f"{title}\n\n" if title else ""
    return f"{header}{record.instructions}".strip()


async def evaluate_record(
    *,
    record: HarveyLabRecord,
    agent: HarveyLabAgent,
    judge: CriterionJudge,
    max_deliverable_chars: int = DEFAULT_MAX_DELIVERABLE_CHARS,
) -> dict[str, Any]:
    """Run the agent on ``record`` and grade it, returning a per-case result.

    The returned dict always carries ``task_id`` + ``practice_area`` and a
    ``kind`` in ``{"scored", "unscoreable"}``. A ``scored`` result also carries
    ``all_pass`` / ``criterion_pass_rate`` / ``n_passed`` / ``n_total`` and the
    agent's ``final_answer`` + produced ``deliverables``.
    """
    output = await agent.forward(record=record)
    criteria_payload = [
        {"id": c.id, "title": c.title, "match_criteria": c.match_criteria, "deliverables": list(c.deliverables)}
        for c in record.criteria
    ]
    has_scoreable = any(str(c.get("match_criteria") or "").strip() for c in criteria_payload)
    base = {"task_id": record.task_id, "practice_area": record.practice_area}
    if not has_scoreable:
        return {**base, "kind": "unscoreable", "final_answer": output.final_answer}
    scored = await asyncio.to_thread(
        score_all_pass,
        criteria=criteria_payload,
        deliverables=output.deliverables,
        task_description=_task_description(record),
        judge=judge,
        max_deliverable_chars=max_deliverable_chars,
    )
    return {
        **base,
        "kind": "scored",
        ALL_PASS_FIELD: scored["all_pass"],
        CRITERION_PASS_RATE_FIELD: scored["criterion_pass_rate"],
        "n_passed": scored["n_passed"],
        "n_total": scored["n_total"],
        "deliverables_produced": sorted(output.deliverables),
        "final_answer": output.final_answer,
    }


async def evaluate_agent_on_records(
    *,
    agent: HarveyLabAgent,
    records: Sequence[HarveyLabRecord],
    judge: CriterionJudge,
    max_deliverable_chars: int = DEFAULT_MAX_DELIVERABLE_CHARS,
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
                    max_deliverable_chars=max_deliverable_chars,
                )
            except Exception as exc:  # noqa: BLE001 - one bad case must not abort the batch
                logger.exception("evaluate_record raised for task %s", record.task_id)
                return {
                    "task_id": record.task_id,
                    "practice_area": record.practice_area,
                    "kind": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }

    per_case = await asyncio.gather(*[_run_one(record) for record in records])

    all_pass_total = 0.0
    rate_total = 0.0
    # Denominator = scored + errored. An errored case counts as 0 (a real
    # failure must deflate, never inflate). An unscoreable case (empty rubric)
    # is excluded entirely — it is not measurable.
    denominator = 0
    num_scored = 0
    num_errored = 0
    num_unscoreable = 0
    for result in per_case:
        kind = result.get("kind")
        if kind == "error":
            num_errored += 1
            denominator += 1  # denominator grows, totals stay 0
        elif kind == "unscoreable":
            num_unscoreable += 1
        else:
            num_scored += 1
            denominator += 1
            all_pass_total += float(result.get(ALL_PASS_FIELD, 0.0))
            rate_total += float(result.get(CRITERION_PASS_RATE_FIELD, 0.0))

    return EvalReport(
        num_cases=len(per_case),
        all_pass=(all_pass_total / denominator) if denominator else 0.0,
        criterion_pass_rate=(rate_total / denominator) if denominator else 0.0,
        num_scored=num_scored,
        num_errored=num_errored,
        num_unscoreable=num_unscoreable,
        per_case=list(per_case),
    )


def run_evaluation(
    *,
    agent: HarveyLabAgent,
    records: Sequence[HarveyLabRecord],
    judge: CriterionJudge,
    config: HarveyLabConfig | None = None,
    max_concurrency: int = 4,
) -> EvalReport:
    """Synchronous wrapper around :func:`evaluate_agent_on_records`."""
    cfg = config or HarveyLabConfig()
    return asyncio.run(
        evaluate_agent_on_records(
            agent=agent,
            records=records,
            judge=judge,
            max_deliverable_chars=cfg.max_deliverable_chars,
            max_concurrency=max_concurrency,
        )
    )


__all__ = [
    "EvalReport",
    "evaluate_agent_on_records",
    "evaluate_record",
    "run_evaluation",
]
