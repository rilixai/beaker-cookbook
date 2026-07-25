"""Local evaluation for the HotpotQA agent (scoring + batch eval)."""

from .local_eval import (
    EvalReport,
    evaluate_agent_on_records,
    evaluate_record,
    run_agent_on_record,
    run_evaluation,
)
from .report import eval_summary, write_json
from .scoring import (
    ANSWER_F1_FIELD,
    ANSWER_FIELD,
    HOTPOTQA_FIELD_WEIGHTS,
    SUPPORTING_TITLES_RECALL_FIELD,
    objective_score,
    score_prediction,
    supporting_titles_recall,
)


__all__ = [
    "ANSWER_F1_FIELD",
    "ANSWER_FIELD",
    "HOTPOTQA_FIELD_WEIGHTS",
    "SUPPORTING_TITLES_RECALL_FIELD",
    "EvalReport",
    "eval_summary",
    "evaluate_agent_on_records",
    "evaluate_record",
    "objective_score",
    "run_agent_on_record",
    "run_evaluation",
    "score_prediction",
    "supporting_titles_recall",
    "write_json",
]
