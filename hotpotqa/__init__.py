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
* :mod:`hotpotqa.rilixai_spec` — the whole GEPA-facing integration
  (``@spec`` runner + ``@spec`` metrics calculator).
* :mod:`hotpotqa.metrics` — the run-metrics trajectory builder.
* :mod:`hotpotqa.feedback` — :class:`HotpotQAFeedback`, the
  per-component reflection narratives.
* :mod:`hotpotqa.config` — :class:`HotpotQAConfig` the runner reads.
"""

from .agent.prompts import (
    DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT,
    DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT,
)
from .agent.types import AgentToolCall, HotpotQAAgentOutput
from .config import HotpotQAConfig
from .data.dataset import (
    HotpotQARecord,
    cases_from_records,
    load_hotpotqa_paper_split,
    load_hotpotqa_split,
    record_to_case,
)
from .data.eval import exact_match_score, f1_score, f1_score_components, normalize_answer
from .feedback import HotpotQAFeedback
from .metrics import build_agent_run_metrics
from .rilixai_spec import (
    HotpotQAMetrics,
    HotpotQARunner,
    HotpotQASandboxConfig,
)


__all__ = [
    "AgentToolCall",
    "DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT",
    "DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT",
    "HotpotQAAgentOutput",
    "HotpotQAConfig",
    "HotpotQAFeedback",
    "HotpotQAMetrics",
    "HotpotQARecord",
    "HotpotQARunner",
    "HotpotQASandboxConfig",
    "build_agent_run_metrics",
    "cases_from_records",
    "exact_match_score",
    "f1_score",
    "f1_score_components",
    "load_hotpotqa_paper_split",
    "load_hotpotqa_split",
    "normalize_answer",
    "record_to_case",
]
