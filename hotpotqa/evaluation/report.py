"""Serialization helpers for the local eval CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .local_eval import EvalReport


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
        "objective": report.objective,
        "field_accuracies": report.field_accuracies,
        "field_sample_counts": report.field_sample_counts,
    }


__all__ = ["eval_summary", "write_json"]
