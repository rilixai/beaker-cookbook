"""Shared config + runtime dispatcher for the HotpotQA agent benchmark.

Wraps the PydanticAI tool-using agent (``.agent.agent.HotpotQAPydanticAgent``)
as an async ``ExtractionRuntime`` the rilixai adapter can drive.
Same retrieve / summarize tool surface as the GEPA paper's HotpotQA
program, but expressed as an idiomatic agent rather than a fixed-hop
DSPy workflow.

``trace_evidence.per_component_feedback`` carries paper-style
per-component diagnostics (``policy_prompt`` + ``summarize_prompt``)
the rilixai adapter surfaces to the reflection LM as the ``Feedback``
field on each component's reflection record.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from rilixai.prompt_optimization.models import PromptCandidate

from .dataset import HotpotQARecord


logger = logging.getLogger(__name__)


RetrievalMode = Literal["distractor", "fullwiki"]


@dataclass(frozen=True)
class HotpotQAPipelineConfig:
    """Knobs for the HotpotQA agent runtime.

    ``retrieval_mode`` chooses the retrieval corpus the agent sees.
    ``fullwiki`` mirrors the GEPA paper's setup (bm25s over the 2017
    Wikipedia abstracts); ``distractor`` is the HF
    ``hotpot_qa[distractor]`` 10-paragraph-per-case shape, useful as a
    test-friendly opt-out.

    ``retrieve_k`` is the number of paragraphs returned per retrieval
    call. The paper uses ``k=7``.

    ``max_iters`` caps the agent-loop length;
    ``pydantic_agent_model`` is the PydanticAI model spec when no
    pre-built agent is supplied.
    """

    retrieval_mode: RetrievalMode = "fullwiki"
    retrieve_k: int = 7
    max_paragraph_chars: int = 1_500
    max_iters: int = 8
    pydantic_agent_model: str | None = None
    # Pinned at ``0.0`` for reproducibility — used by the PydanticAI
    # agent's outer model settings and its raw summarize-tool call.
    pydantic_agent_temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.retrieve_k < 1:
            raise ValueError(f"HotpotQAPipelineConfig.retrieve_k must be >= 1, got {self.retrieve_k}.")
        if self.max_iters < 1:
            raise ValueError(f"HotpotQAPipelineConfig.max_iters must be >= 1, got {self.max_iters}.")
        if self.retrieval_mode not in ("distractor", "fullwiki"):
            raise ValueError(
                f"HotpotQAPipelineConfig.retrieval_mode must be one of "
                f"'distractor' / 'fullwiki', got {self.retrieval_mode!r}."
            )


@dataclass
class HotpotQARunResult:
    """One case's output, shaped for the rilixai field-extractor contract."""

    answer: str
    retrieved_titles: list[str]
    run_metrics: dict[str, Any] = field(default_factory=dict)


def build_hotpotqa_runtime(
    *,
    config: HotpotQAPipelineConfig | None = None,
    pydantic_agent: Any | None = None,
) -> Callable[..., Awaitable[HotpotQARunResult]]:
    """Build an async ``ExtractionRuntime`` for the HotpotQA agent."""
    cfg = config or HotpotQAPipelineConfig()
    from .agent.runtime import build_pydantic_agent_runtime

    return build_pydantic_agent_runtime(cfg=cfg, agent=pydantic_agent)


# ─── Shared helpers used by the agent runtime ───────────────────────────


def _unpack_runtime_inputs(kwargs: dict[str, Any]) -> tuple[HotpotQARecord, PromptCandidate]:
    record = kwargs.get("input")
    if not isinstance(record, HotpotQARecord):
        raise TypeError(f"HotpotQA runtime expected `input` to be a HotpotQARecord, got {type(record).__name__}.")
    candidate = kwargs.get("candidate")
    if not isinstance(candidate, PromptCandidate):
        raise TypeError(
            f"HotpotQA runtime expected `candidate` to be a PromptCandidate, got {type(candidate).__name__}."
        )
    return record, candidate


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"
