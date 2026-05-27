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
"""

from .agent.feedback import build_agent_per_component_feedback
from .agent.prompts import (
    DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT,
    DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT,
    hotpotqa_pydantic_agent_seed_candidate,
)
from .agent.types import AgentToolCall, HotpotQAAgentOutput
from .dataset import (
    HotpotQARecord,
    cases_from_records,
    load_hotpotqa_paper_split,
    load_hotpotqa_split,
    record_to_case,
)
from .hotpot_eval import exact_match_score, f1_score, f1_score_components, normalize_answer
from .metrics import (
    ANSWER_F1_FIELD,
    ANSWER_FIELD,
    HOTPOTQA_FIELD_WEIGHTS,
    SUPPORTING_TITLES_RECALL_FIELD,
    HotpotQAFieldConfig,
    HotpotQAMetricsCalculator,
    build_hotpotqa_field_extractor,
)
from .pipeline import (
    HotpotQAPipelineConfig,
    HotpotQARunResult,
    build_hotpotqa_runtime,
)
from .spec import build_hotpotqa_spec


__all__ = [
    "ANSWER_F1_FIELD",
    "ANSWER_FIELD",
    "AgentToolCall",
    "DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT",
    "DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT",
    "HOTPOTQA_FIELD_WEIGHTS",
    "HotpotQAAgentOutput",
    "HotpotQAFieldConfig",
    "HotpotQAMetricsCalculator",
    "HotpotQAPipelineConfig",
    "HotpotQARecord",
    "HotpotQARunResult",
    "SUPPORTING_TITLES_RECALL_FIELD",
    "build_agent_per_component_feedback",
    "build_hotpotqa_field_extractor",
    "build_hotpotqa_runtime",
    "build_hotpotqa_spec",
    "cases_from_records",
    "exact_match_score",
    "f1_score",
    "f1_score_components",
    "hotpotqa_pydantic_agent_seed_candidate",
    "load_hotpotqa_paper_split",
    "load_hotpotqa_split",
    "normalize_answer",
    "record_to_case",
]
