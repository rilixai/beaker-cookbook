"""Serialization helpers for the local eval CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .local_eval import EvalReport
from .scoring import ALL_PASS_FIELD


def write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as pretty JSON, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def eval_summary(report: EvalReport, *, split: str) -> dict[str, Any]:
    """Build the ``eval_summary.json`` payload from an eval report."""
    return {
        "split": split,
        "num_cases": report.num_cases,
        "num_scored": report.num_scored,
        "num_errored": report.num_errored,
        "num_unscoreable": report.num_unscoreable,
        "all_pass": report.all_pass,
        "criterion_pass_rate": report.criterion_pass_rate,
    }


def heldout_subset_summary(per_case: list[dict[str, Any]], excluded_area_ids: set[str]) -> dict[str, Any]:
    """Summarize the measurable cases whose practice area is outside ``excluded_area_ids``.

    ``excluded_area_ids`` are the practice areas reserved for the validation
    pool; the held-out subset drops them so the mean is over disjoint domains.
    Mirrors the headline aggregation: scored + errored cases are measurable
    (an errored case counts as ``0``), while unscoreable (empty-rubric) cases
    are excluded entirely.
    """
    held = [
        r
        for r in per_case
        if r.get("kind") in ("scored", "error") and str(r.get("practice_area") or "") not in excluded_area_ids
    ]
    # Errored held-out cases contribute 0 (a real failure must deflate).
    scores = [float(r.get(ALL_PASS_FIELD, 0.0)) for r in held]
    return {
        "excluded_practice_areas": sorted(excluded_area_ids),
        "num_heldout_cases": len(held),
        f"{ALL_PASS_FIELD}_heldout": (sum(scores) / len(scores)) if scores else None,
        "note": (
            f"{ALL_PASS_FIELD} is over ALL measurable cases (scored + errored) incl. the "
            f"reserved cross-area validation pool; {ALL_PASS_FIELD}_heldout is the subset "
            f"whose practice areas fall outside that pool."
        ),
    }


__all__ = ["eval_summary", "heldout_subset_summary", "write_json"]
