"""HotpotQA — a PydanticAI tool-using agent + a local evaluation.

The multi-hop QA task wrapped as the kind of agent you'd actually write
in PydanticAI: two tools
(``retrieve_k(query)`` + ``summarize(question, passages, context=None)``)
and a structured Pydantic answer terminator, steered by two prompts
(``policy_prompt`` + ``summarize_prompt``).

Retrieval is pluggable: ``fullwiki`` (open-domain — bm25s over the
2017 Wikipedia abstracts dump) or ``distractor`` (HF
``hotpot_qa[distractor]`` 10-paragraph corpus, the cheap option for
tests).

Layout (each subpackage groups one concern):

* :mod:`hotpotqa.agent` — PydanticAI agent internals (incl. its
  ``retrieve_k`` tool implementation under :mod:`.agent.retrieval`).
* :mod:`hotpotqa.data` — raw HotpotQA primitives (dataset loader,
  official scorer, supporting-fact helpers).
* :mod:`hotpotqa.evaluation` — scoring + the bounded-concurrency batch
  evaluator and its JSON reports.
* :mod:`hotpotqa.config` — :class:`HotpotQAConfig` shared by the CLI,
  the agent, and the evaluation.
* :mod:`hotpotqa.cli` — command-line entry point.
"""

from .agent.prompts import (
    DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT,
    DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT,
    hotpotqa_default_prompts,
)
from .agent.types import AgentToolCall, HotpotQAAgentOutput
from .config import HotpotQAConfig
from .data.dataset import (
    HotpotQAParagraph,
    HotpotQARecord,
    load_hotpotqa_paper_split,
    load_hotpotqa_split,
    records_from_raw,
)
from .data.eval import exact_match_score, f1_score, f1_score_components, normalize_answer
from .evaluation.local_eval import (
    EvalReport,
    evaluate_agent_on_records,
    evaluate_record,
    run_agent_on_record,
    run_evaluation,
)
from .evaluation.report import eval_summary, write_json
from .evaluation.scoring import (
    ANSWER_F1_FIELD,
    ANSWER_FIELD,
    HOTPOTQA_FIELD_WEIGHTS,
    SUPPORTING_TITLES_RECALL_FIELD,
    objective_score,
    score_prediction,
)


__all__ = [
    "ANSWER_F1_FIELD",
    "ANSWER_FIELD",
    "DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT",
    "DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT",
    "HOTPOTQA_FIELD_WEIGHTS",
    "SUPPORTING_TITLES_RECALL_FIELD",
    "AgentToolCall",
    "EvalReport",
    "HotpotQAAgentOutput",
    "HotpotQAConfig",
    "HotpotQAParagraph",
    "HotpotQARecord",
    "eval_summary",
    "evaluate_agent_on_records",
    "evaluate_record",
    "exact_match_score",
    "f1_score",
    "f1_score_components",
    "hotpotqa_default_prompts",
    "load_hotpotqa_paper_split",
    "load_hotpotqa_split",
    "normalize_answer",
    "objective_score",
    "records_from_raw",
    "run_agent_on_record",
    "run_evaluation",
    "score_prediction",
    "write_json",
]
