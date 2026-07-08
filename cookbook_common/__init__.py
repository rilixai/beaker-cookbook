"""Shared, recipe-agnostic helpers for the cookbook benchmark recipes.

The individual recipes (``apex_agents``, ``hotpotqa``) are self-contained, but a
few pieces are genuinely identical across them — the SDK-only local evaluation
loop and the local CLI plumbing (candidate-JSON loading, spec-validation
logging, eval-report serialization). They live here so the recipes import one
implementation instead of copying it.
"""

from __future__ import annotations

from .cli_support import (
    eval_summary,
    load_targets_from_json,
    validate_and_log,
    write_eval_report,
    write_json,
)
from .local_eval import (
    LocalEvalReport,
    evaluate_targets_on_cases,
    run_local_evaluation,
)


__all__ = [
    "LocalEvalReport",
    "eval_summary",
    "evaluate_targets_on_cases",
    "load_targets_from_json",
    "run_local_evaluation",
    "validate_and_log",
    "write_eval_report",
    "write_json",
]
