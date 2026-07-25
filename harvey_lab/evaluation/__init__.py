"""Local rubric evaluation for the Harvey LAB agent (judge + batch eval)."""

from .local_eval import (
    EvalReport,
    evaluate_agent_on_records,
    evaluate_record,
    run_evaluation,
)
from .report import eval_summary, heldout_subset_summary, write_json
from .scoring import (
    ALL_PASS_FIELD,
    CRITERION_PASS_RATE_FIELD,
    CriterionJudge,
    build_criterion_judge,
    score_all_pass,
)


__all__ = [
    "ALL_PASS_FIELD",
    "CRITERION_PASS_RATE_FIELD",
    "CriterionJudge",
    "EvalReport",
    "build_criterion_judge",
    "eval_summary",
    "evaluate_agent_on_records",
    "evaluate_record",
    "heldout_subset_summary",
    "run_evaluation",
    "score_all_pass",
    "write_json",
]
