"""Shared ``cli.py`` plumbing for the cookbook recipes.

The local CLI (``validate`` / ``evaluate``) needs candidate-JSON loading,
spec-validation logging, and eval-report serialization. This module holds that
recipe-agnostic glue; the only recipe-specific inputs are passed in as arguments
(the seed targets and the spec/logger). Each recipe keeps its own copy so the
folder is self-contained.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rilixai import OptimizationTargets, Spec, optimization_targets_from_prompts, validate_spec

from .local_eval import LocalEvalReport


def write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as pretty JSON, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def load_targets_from_json(path: Path | None, *, seed_targets: OptimizationTargets) -> OptimizationTargets:
    """Load an ``evaluate --candidate-json`` file into ``OptimizationTargets``.

    Returns ``seed_targets`` unchanged when ``path`` is ``None``. Otherwise
    accepts the ``OptimizationTargets`` wire shape (``{"prompts": {...}}``), the
    legacy ``PromptCandidate`` shape (``{"components": {...}}``) written by the
    pre-migration optimizer, or a bare ``{name: text}`` mapping.
    """
    if path is None:
        return seed_targets
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "prompts" in raw:
        prompts = raw["prompts"]
    elif isinstance(raw, dict) and "components" in raw:
        prompts = raw["components"]
    else:
        prompts = raw
    if not isinstance(prompts, dict):
        raise ValueError(f"Candidate JSON at {path} must be an object of prompt name → text.")
    parsed = {str(k): str(v) for k, v in prompts.items()}
    # Guard against a mis-shaped/typo'd file being read as a bare name→text map:
    # ``apply_candidate`` silently ignores unknown component names, so without
    # this a candidate whose keys match nothing would evaluate the *seed*
    # prompts and report that score as the candidate's.
    known = set(seed_targets.to_dict())
    if not (parsed.keys() & known):
        raise ValueError(
            f"Candidate JSON at {path} has no recognized prompt components "
            f"(expected any of {sorted(known)}, got {sorted(parsed)})."
        )
    return optimization_targets_from_prompts(parsed)


def validate_and_log(spec: Spec, *, logger: logging.Logger) -> int:
    """Run ``validate_spec`` and log the seed prompt summary; return exit code 0."""
    validate_spec(spec)
    logger.info(
        "Spec %r validated: %d seed prompt(s) %s.",
        spec.name,
        len(spec.seed_targets.prompts),
        sorted(spec.seed_targets.prompts),
    )
    return 0


def eval_summary(report: LocalEvalReport, *, split: str) -> dict[str, Any]:
    """Build the ``eval_summary.json`` payload from a local-eval report."""
    return {
        "split": split,
        "num_cases": report.num_cases,
        "num_errored": report.num_errored,
        "num_unscoreable": report.num_unscoreable,
        "objective": report.objective,
        "field_accuracies": report.field_accuracies,
        "field_sample_counts": report.field_sample_counts,
    }


def write_eval_report(report: LocalEvalReport, *, output_dir: Path, split: str) -> None:
    """Write the summary + per-case outputs JSON files for an eval run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "eval_summary.json", eval_summary(report, split=split))
    write_json(output_dir / "eval_outputs.json", report.per_case)


__all__ = [
    "eval_summary",
    "load_targets_from_json",
    "validate_and_log",
    "write_eval_report",
    "write_json",
]
