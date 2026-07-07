"""HotpotQA — PydanticAI tool-using agent + rilixai GEPA optimization.

The multi-hop QA task wrapped as the kind of agent a customer would
actually write in PydanticAI: two tools
(``retrieve_k(query)`` + ``summarize(question, passages, context=None)``)
and a structured Pydantic answer terminator. Two optimizable
components — ``policy_prompt`` + ``summarize_prompt`` — get rewritten
by rilixai's GEPA loop.

Retrieval is pluggable: ``fullwiki`` (paper parity — bm25s over the
2017 Wikipedia abstracts dump) or ``distractor`` (HF
``hotpot_qa[distractor]`` 10-paragraph corpus, opt-out for tests).

Layout (each subpackage groups one concern):

* :mod:`hotpotqa.agent` — PydanticAI agent internals (incl. its
  ``retrieve_k`` tool implementation under :mod:`.agent.retrieval`).
* :mod:`hotpotqa.data` — raw HotpotQA primitives (dataset loader,
  official scorer, supporting-fact helpers).
* :mod:`hotpotqa.optimization` — GEPA-facing surface (spec, runtime
  adapter, metrics aggregator, per-component feedback strings).
* :mod:`hotpotqa.config` — :class:`HotpotQAConfig` shared by the CLI
  and the runtime.
* :mod:`hotpotqa.cli` — command-line entry point.
"""

from .agent.prompts import (
    DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT,
    DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT,
    hotpotqa_pydantic_agent_seed_targets,
)
from .agent.types import AgentToolCall, HotpotQAAgentOutput
from .config import HotpotQAConfig
from .data.dataset import (
    HotpotQADataLoader,
    HotpotQARecord,
    cases_from_records,
    load_hotpotqa_paper_split,
    load_hotpotqa_split,
    record_to_case,
)
from .data.eval import exact_match_score, f1_score, f1_score_components, normalize_answer
from .optimization.feedback import build_agent_per_component_feedback
from .optimization.local_eval import LocalEvalReport, run_local_evaluation
from .optimization.metrics import (
    ANSWER_F1_FIELD,
    ANSWER_FIELD,
    HOTPOTQA_FIELD_WEIGHTS,
    SUPPORTING_TITLES_RECALL_FIELD,
    HotpotQAScorer,
)
from .optimization.runtime import build_hotpotqa_run_case
from .optimization.spec import build_hotpotqa_spec


__all__ = [
    "ANSWER_F1_FIELD",
    "ANSWER_FIELD",
    "AgentToolCall",
    "DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT",
    "DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT",
    "HOTPOTQA_FIELD_WEIGHTS",
    "HotpotQAAgentOutput",
    "HotpotQAConfig",
    "HotpotQADataLoader",
    "HotpotQARecord",
    "HotpotQAScorer",
    "LocalEvalReport",
    "SUPPORTING_TITLES_RECALL_FIELD",
    "build_agent_per_component_feedback",
    "build_hotpotqa_run_case",
    "build_hotpotqa_spec",
    "cases_from_records",
    "exact_match_score",
    "f1_score",
    "f1_score_components",
    "hotpotqa_pydantic_agent_seed_targets",
    "load_hotpotqa_paper_split",
    "load_hotpotqa_split",
    "normalize_answer",
    "record_to_case",
    "run_local_evaluation",
]
