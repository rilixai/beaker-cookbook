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
    hotpotqa_pydantic_agent_seed_candidate,
)
from .agent.types import AgentToolCall, HotpotQAAgentOutput
from .config import HotpotQAConfig
from .data.dataset import (
    HotpotQARecord,
    cases_from_records,
    load_hotpotqa_paper_split,
    load_hotpotqa_split,
    record_to_sample,
)
from .data.eval import exact_match_score, f1_score, f1_score_components, normalize_answer
from .optimization.feedback import build_agent_per_component_feedback
from .optimization.runtime import build_agent_run_metrics
from .optimization.spec import (
    HotpotQAMetrics,
    HotpotQARunner,
    HotpotQASandboxConfig,
    build_hotpotqa_spec,
)


__all__ = [
    "AgentToolCall",
    "DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT",
    "DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT",
    "HotpotQAAgentOutput",
    "HotpotQAConfig",
    "HotpotQAMetrics",
    "HotpotQARecord",
    "HotpotQARunner",
    "HotpotQASandboxConfig",
    "build_agent_per_component_feedback",
    "build_agent_run_metrics",
    "build_hotpotqa_spec",
    "cases_from_records",
    "exact_match_score",
    "f1_score",
    "f1_score_components",
    "hotpotqa_pydantic_agent_seed_candidate",
    "load_hotpotqa_paper_split",
    "load_hotpotqa_split",
    "normalize_answer",
    "record_to_sample",
]
