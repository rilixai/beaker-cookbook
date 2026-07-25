"""SDK-only local evaluation loop (Shape B) for the HotpotQA recipe.

The full GEPA reflect/propose loop lives in the optional ``rilixai-runtime``
package and runs server-side for hosted ``rilixai run`` triggers. The
lightweight ``rilixai`` SDK ships only the contract types, so this module
provides the offline counterpart that backs ``cli.py evaluate``: run every
:class:`~rilixai.Case` through the spec's ``run_case`` on ONE fixed
:class:`~rilixai.OptimizationTargets` bundle and aggregate the spec scorer's
per-field scores + objective. It mirrors what the hosted runner does when it
scores a single candidate, without pulling in the optimizer engine.

The loop only touches the ``rilixai`` contract types, so it stays wholly
inside this recipe (no shared cookbook package).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rilixai import Case, OptimizationTargets, Spec


logger = logging.getLogger(__name__)


@dataclass
class LocalEvalReport:
    """Aggregate of one candidate's scores over a set of cases."""

    num_cases: int
    objective: float
    field_accuracies: dict[str, float]
    field_sample_counts: dict[str, int]
    per_case: list[dict[str, Any]] = field(default_factory=list)
    num_errored: int = 0
    num_unscoreable: int = 0


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

    async def _run_one(case: Case) -> tuple[Case, Any, Exception | None]:
        async with semaphore:
            try:
                result = await spec.run_case(case=case, targets=targets, runtime=None)
                score = await spec.scorer.score_case(case=case, result=result)
                return case, score, None
            except Exception as exc:  # noqa: BLE001 - one bad case must not abort the batch
                # Contain per-case failures: score the case ``0`` (below)
                # instead of letting the exception propagate out of
                # ``asyncio.gather`` and discard every already-completed
                # result. Errored cases contribute zero so accuracy cannot be
                # inflated.
                logger.exception("run_case/score_case raised for case %s", case.case_id)
                return case, None, exc

    results = await asyncio.gather(*[_run_one(case) for case in cases])

    # First pass: the union of field names any successful case scored. Errored
    # cases are then scored ``0`` on every one of these fields so a failure
    # deflates (never inflates) both the objective and the field accuracies.
    field_names: set[str] = set()
    for _case, score, err in results:
        if err is None:
            field_names.update(score.field_scores)

    objective_total = 0.0
    # Denominator for the objective mean. An errored case counts as 0 (a real
    # failure must deflate, never inflate). A successful-but-unscoreable case
    # (empty ``field_scores`` — e.g. an empty rubric) is excluded entirely, the
    # same way it is dropped from the field accuracies: it is not measurable, so
    # it must not drag the objective below field_accuracies.
    objective_count = 0
    field_totals: dict[str, float] = dict.fromkeys(field_names, 0.0)
    field_counts: dict[str, int] = dict.fromkeys(field_names, 0)
    per_case: list[dict[str, Any]] = []
    num_errored = 0
    num_unscoreable = 0
    for case, score, err in results:
        if err is not None:
            num_errored += 1
            objective_count += 1  # denominator grows, total stays 0
            for name in field_names:
                field_counts[name] += 1  # denominator grows, total stays 0
            per_case.append(
                {
                    "case_id": case.case_id,
                    "group_key": case.group_key,
                    "objective": 0.0,
                    "field_scores": {},
                    "error": f"{type(err).__name__}: {err}",
                }
            )
            continue
        if not score.field_scores:
            # Nothing was scoreable (e.g. empty rubric): omit from both the
            # objective mean and the field accuracies.
            num_unscoreable += 1
            per_case.append(
                {
                    "case_id": case.case_id,
                    "group_key": case.group_key,
                    "objective": float(score.objective),
                    "field_scores": {},
                    "unscoreable": True,
                }
            )
            continue
        objective_total += float(score.objective)
        objective_count += 1
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

    num_cases = len(results)
    field_accuracies = {name: field_totals[name] / field_counts[name] for name in field_totals if field_counts[name]}
    return LocalEvalReport(
        num_cases=num_cases,
        objective=(objective_total / objective_count) if objective_count else 0.0,
        field_accuracies=field_accuracies,
        field_sample_counts=dict(field_counts),
        per_case=per_case,
        num_errored=num_errored,
        num_unscoreable=num_unscoreable,
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
