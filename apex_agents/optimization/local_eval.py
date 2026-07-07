"""SDK-only local evaluation loop (Shape B).

The full GEPA reflect/propose loop lives in the optional ``rilixai-runtime``
package and runs server-side for hosted ``rilixai run`` triggers. The
lightweight ``rilixai`` SDK ships only the contract types, so this module
provides the offline counterpart: run every :class:`~rilixai.Case` through the
spec's ``run_case`` on ONE fixed :class:`~rilixai.OptimizationTargets` bundle
and aggregate the :class:`ApexAgentsScorer`'s per-field scores + objective.

It backs ``cli.py evaluate`` and mirrors what the hosted runner does when it
scores a single candidate, without pulling in the optimizer engine.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rilixai import Case, OptimizationTargets, Spec


@dataclass
class LocalEvalReport:
    """Aggregate of one candidate's scores over a set of cases."""

    num_cases: int
    objective: float
    field_accuracies: dict[str, float]
    field_sample_counts: dict[str, int]
    per_case: list[dict[str, Any]] = field(default_factory=list)


async def evaluate_targets_on_cases(
    *,
    spec: Spec,
    targets: OptimizationTargets,
    cases: Sequence[Case],
    max_concurrency: int = 4,
) -> LocalEvalReport:
    """Score ``targets`` over ``cases`` via ``spec.run_case`` + ``spec.scorer``.

    Runs cases concurrently (bounded by ``max_concurrency``). The mean of the
    per-case objectives is the report objective; per-field means give the
    field accuracy table.
    """
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def _run_one(case: Case) -> tuple[Case, Any]:
        async with semaphore:
            result = await spec.run_case(case=case, targets=targets, runtime=None)
            score = await spec.scorer.score_case(case=case, result=result)
            return case, score

    pairs = await asyncio.gather(*[_run_one(case) for case in cases])

    objective_total = 0.0
    field_totals: dict[str, float] = {}
    field_counts: dict[str, int] = {}
    per_case: list[dict[str, Any]] = []
    for case, score in pairs:
        objective_total += float(score.objective)
        for name, value in score.field_scores.items():
            field_totals[name] = field_totals.get(name, 0.0) + float(value)
            field_counts[name] = field_counts.get(name, 0) + 1
        per_case.append(
            {
                "case_id": case.case_id,
                "group_key": case.group_key,
                "objective": float(score.objective),
                "field_scores": dict(score.field_scores),
            }
        )

    num_cases = len(pairs)
    field_accuracies = {name: total / field_counts[name] for name, total in field_totals.items()}
    return LocalEvalReport(
        num_cases=num_cases,
        objective=(objective_total / num_cases) if num_cases else 0.0,
        field_accuracies=field_accuracies,
        field_sample_counts=dict(field_counts),
        per_case=per_case,
    )


def run_local_evaluation(
    *,
    spec: Spec,
    targets: OptimizationTargets,
    cases: Sequence[Case],
    max_concurrency: int = 4,
) -> LocalEvalReport:
    """Synchronous wrapper around :func:`evaluate_targets_on_cases`."""
    return asyncio.run(
        evaluate_targets_on_cases(
            spec=spec,
            targets=targets,
            cases=cases,
            max_concurrency=max_concurrency,
        )
    )


__all__ = ["LocalEvalReport", "evaluate_targets_on_cases", "run_local_evaluation"]
