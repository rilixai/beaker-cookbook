"""Re-export of the shared SDK-only local evaluation loop.

The implementation is recipe-agnostic (it touches only the ``rilixai`` contract
types), so it lives once in :mod:`cookbook_common.local_eval` and is re-exported
here to preserve the ``apex_agents.optimization.local_eval`` import path.
"""

from __future__ import annotations

from cookbook_common.local_eval import (
    LocalEvalReport,
    evaluate_targets_on_cases,
    run_local_evaluation,
)


__all__ = ["LocalEvalReport", "evaluate_targets_on_cases", "run_local_evaluation"]
