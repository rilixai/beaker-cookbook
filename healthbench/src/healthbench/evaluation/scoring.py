"""HealthBench scoring, matching ``openai/simple-evals``.

Per case: ``overall_score = points of met criteria / sum of positive points``.
Negative criteria (e.g. "recommends a harmful dose") subtract when met, so a
case score can go below 0. Each case also scores every tag it carries:

* example tags (``theme:*``, ``physician_agreed_category:*``) get the case's
  overall score;
* rubric tags (``axis:*``, ``level:*``, ``cluster:*``) are re-scored over just
  the criteria carrying that tag.

Across cases, each metric is the mean **clipped to [0, 1]**, with a bootstrap
standard error. A case that errored (agent or judge) scores 0: failures must
deflate, never inflate, the headline number.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from .grader import CriterionGrade


OVERALL = "overall_score"


def rubric_score(grades: Sequence[CriterionGrade]) -> float | None:
    possible = sum(g.points for g in grades if g.points > 0)
    if possible == 0:
        return None
    return sum(g.points for g in grades if g.criteria_met) / possible


def case_metrics(grades: Sequence[CriterionGrade], example_tags: Sequence[str]) -> dict[str, float]:
    overall = rubric_score(grades)
    if overall is None:
        raise ValueError("Rubric has no positive-point criteria.")
    metrics = {OVERALL: overall}
    metrics.update(dict.fromkeys(example_tags, overall))
    by_tag: dict[str, list[CriterionGrade]] = defaultdict(list)
    for grade in grades:
        for tag in set(grade.tags):
            by_tag[tag].append(grade)
    for tag, tagged in by_tag.items():
        score = rubric_score(tagged)
        if score is not None:  # tag has at least one positive criterion
            metrics[tag] = score
    return metrics


def _clipped_mean(values: Sequence[float]) -> float:
    return min(1.0, max(0.0, sum(values) / len(values)))


def aggregate(per_case: Sequence[dict[str, float]], *, bootstrap: int = 1000, seed: int = 0) -> dict[str, Any]:
    """``{metric: {"mean", "bootstrap_std", "n_samples"}}`` over cases that carry it."""
    values: dict[str, list[float]] = defaultdict(list)
    for metrics in per_case:
        for key, value in metrics.items():
            values[key].append(value)
    rng = random.Random(seed)
    out: dict[str, Any] = {}
    for key in sorted(values, key=lambda k: (k != OVERALL, k)):
        vals = values[key]
        means = [_clipped_mean(rng.choices(vals, k=len(vals))) for _ in range(bootstrap)]
        center = sum(means) / len(means) if means else 0.0
        std = (sum((m - center) ** 2 for m in means) / len(means)) ** 0.5 if means else 0.0
        out[key] = {"mean": _clipped_mean(vals), "bootstrap_std": std, "n_samples": len(vals)}
    return out
