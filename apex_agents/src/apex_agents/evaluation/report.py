"""Serialization helpers for the local eval CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .local_eval import EvalReport
from .scoring import RUBRIC_FIELD


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
        RUBRIC_FIELD: report.rubric_pass_rate,
    }


def heldout_subset_summary(per_case: list[dict[str, Any]], excluded_world_ids: set[str]) -> dict[str, Any]:
    """Summarize the measurable cases whose world is outside ``excluded_world_ids``.

    ``excluded_world_ids`` are the worlds reserved for the validation pool;
    the held-out subset drops them so the mean is over disjoint worlds.
    Mirrors the headline aggregation: scored + errored cases are measurable
    (an errored case counts as ``0``), while unscoreable (empty-rubric) cases
    are excluded entirely.
    """
    held = [
        r
        for r in per_case
        if r.get("kind") in ("scored", "error") and str(r.get("world_id") or "") not in excluded_world_ids
    ]
    # Errored held-out cases contribute 0 (a real failure must deflate).
    scores = [float(r.get(RUBRIC_FIELD, 0.0)) for r in held]
    return {
        "excluded_world_ids": sorted(excluded_world_ids),
        "num_heldout_cases": len(held),
        f"{RUBRIC_FIELD}_heldout": (sum(scores) / len(scores)) if scores else None,
        "note": (
            f"{RUBRIC_FIELD} is over ALL measurable cases (scored + errored) incl. the "
            f"reserved cross-world validation pool; {RUBRIC_FIELD}_heldout is the subset "
            f"whose worlds fall outside that pool."
        ),
    }


__all__ = ["eval_summary", "heldout_subset_summary", "write_json"]
