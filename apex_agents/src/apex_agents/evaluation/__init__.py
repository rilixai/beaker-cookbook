"""Local evaluation of the APEX-Agents agent.

* :mod:`.scoring` — the per-criterion LLM judge and the ``rubric_pass_rate``
  aggregation for a single task.
* :mod:`.local_eval` — the bounded-concurrency batch evaluator that runs the
  agent over a set of records and aggregates the metric.
* :mod:`.report` — the JSON artifacts the CLI writes.
"""

from .local_eval import (
    EvalReport,
    evaluate_agent_on_records,
    evaluate_record,
    run_evaluation,
)
from .report import eval_summary, heldout_subset_summary, write_json
from .scoring import (
    DEFAULT_JUDGE_MODEL,
    RUBRIC_FIELD,
    RubricJudge,
    build_rubric_judge,
    score_rubric,
)


__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "RUBRIC_FIELD",
    "EvalReport",
    "RubricJudge",
    "build_rubric_judge",
    "eval_summary",
    "evaluate_agent_on_records",
    "evaluate_record",
    "heldout_subset_summary",
    "run_evaluation",
    "score_rubric",
    "write_json",
]
