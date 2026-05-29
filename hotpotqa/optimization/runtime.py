"""Async ``ExtractionRuntime`` adapter — the GEPA-facing surface.

Wraps :class:`~hotpotqa.agent.agent.HotpotQAPydanticAgent` as an async
runtime the rilixai adapter can drive. Translates the agent's
variable-length tool-call trajectory into the trajectory-dict schema
the optimizer's adapter expects, and populates
``trace_evidence.per_component_feedback`` with paper-style
per-component diagnostics (``policy_prompt`` + ``summarize_prompt``)
the reflection LM reads when rewriting each component.

This module is the single entry point GEPA hits per case — everything
under :mod:`hotpotqa.agent` is implementation detail it composes.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from rilixai.prompt_optimization.models import PromptCandidate

from ..agent.types import AgentToolCall, HotpotQAAgentOutput
from ..config import HotpotQAConfig
from ..data.dataset import HotpotQARecord
from .feedback import build_agent_per_component_feedback


logger = logging.getLogger(__name__)


@dataclass
class HotpotQARunResult:
    """One case's output, shaped for the rilixai field-extractor contract."""

    answer: str
    retrieved_titles: list[str]
    run_metrics: dict[str, Any] = field(default_factory=dict)


def build_hotpotqa_runtime(
    *,
    config: HotpotQAConfig | None = None,
    pydantic_agent: Any | None = None,
) -> Callable[..., Awaitable[HotpotQARunResult]]:
    """Build an async ``ExtractionRuntime`` for the HotpotQA agent.

    Returns a callable the rilixai adapter invokes per case. ``config``
    pins the retrieval mode + agent loop knobs; ``pydantic_agent`` is
    an optional pre-built :class:`HotpotQAPydanticAgent` (tests inject
    one with a scripted ``FunctionModel`` here).
    """
    cfg = config or HotpotQAConfig()

    # Defer pydantic-ai-touching imports so callers that just need the
    # runtime symbol (e.g. spec construction at module import time)
    # don't pay the import cost until the runtime actually fires.
    from ..agent.agent import HotpotQAPydanticAgent
    from ..agent.retrieval import build_retrieve_k_fn_for_case

    resolved_agent: HotpotQAPydanticAgent
    if isinstance(pydantic_agent, HotpotQAPydanticAgent):
        resolved_agent = pydantic_agent
    elif pydantic_agent is None:
        if cfg.pydantic_agent_model is None:
            raise ValueError(
                "build_hotpotqa_runtime requires either a pre-built agent or HotpotQAConfig.pydantic_agent_model."
            )
        resolved_agent = HotpotQAPydanticAgent(
            model=cfg.pydantic_agent_model,
            # Track the task model on the summarize call too so the two
            # LLM call sites can't drift (the agent's outer model + the
            # raw OpenAI summarize call). ``pydantic_agent_model`` is
            # a PydanticAI spec like ``"openai:gpt-4.1-mini"``; the raw
            # OpenAI call wants the bare ``"gpt-4.1-mini"`` model name.
            summarize_model=_bare_openai_model(cfg.pydantic_agent_model),
            top_k=cfg.retrieve_k,
            max_iters=cfg.max_iters,
            temperature=cfg.pydantic_agent_temperature,
        )
    else:
        raise TypeError(
            f"pydantic_agent must be a HotpotQAPydanticAgent or None, got {type(pydantic_agent).__name__}."
        )

    async def _runtime(**kwargs: Any) -> HotpotQARunResult:
        record, candidate = _unpack_runtime_inputs(kwargs)
        resolved_agent.apply_candidate(candidate.components)
        # Reuse the mode-dispatching retrieve fn so the agent sees the
        # same retrieval corpus the runtime's ``cfg.retrieval_mode``
        # selects. Before this, ``_do_retrieve`` always searched the
        # case's 10 local paragraphs regardless of mode — making
        # nominally-fullwiki runs silently use distractor retrieval.
        retrieve_k_fn = build_retrieve_k_fn_for_case(record=record, cfg=cfg)
        output = await resolved_agent.forward(
            question=record.question,
            paragraphs=record.paragraphs,
            gold_supporting_titles=record.supporting_titles,
            retrieve_k_fn=retrieve_k_fn,
        )
        run_metrics = build_agent_run_metrics(record=record, output=output, agent_kind="pydantic", config=cfg)
        return HotpotQARunResult(
            answer=output.answer,
            retrieved_titles=[p.title for p in output.retrieved_paragraphs],
            run_metrics=run_metrics,
        )

    return _runtime


def build_agent_run_metrics(
    *,
    record: HotpotQARecord,
    output: HotpotQAAgentOutput,
    agent_kind: str,
    config: HotpotQAConfig,
) -> dict[str, Any]:
    """Translate an agent's tool-call trace into rilixai trajectory metadata."""
    gold_titles_lower = {t.strip().lower() for t in record.supporting_titles}
    retrieved_titles_lower = {p.title.strip().lower() for p in output.retrieved_paragraphs}
    missing_gold_titles = sorted(
        title for title in record.supporting_titles if title.strip().lower() not in retrieved_titles_lower
    )
    spurious_titles = sorted(
        p.title for p in output.retrieved_paragraphs if p.title.strip().lower() not in gold_titles_lower
    )

    retrieval_span_candidates: dict[str, str] = {
        p.title: _truncate(p.text, config.max_paragraph_chars) for p in output.retrieved_paragraphs
    }

    tool_calls_detail: list[dict[str, Any]] = []
    retrieval_reasoning: list[str] = []
    documents_remaining_per_hop: list[str] = []
    policy_reasoning: list[str] = []
    tool_counts: dict[str, int] = {}
    for step in output.tool_calls:
        tool_calls_detail.append(_tool_call_for_agent_step(step))
        if step.thought:
            policy_reasoning.append(f"Step {step.step_index + 1}: {step.thought}")
        tool_counts[step.tool_name] = tool_counts.get(step.tool_name, 0) + 1
        if step.tool_name == "retrieve_k" or step.tool_name == "search":
            retrieval_reasoning.append(
                f"{step.tool_name} args={step.tool_args}; "
                f"gold titles still missing after this call: {step.gold_titles_remaining_after}"
            )
            documents_remaining_per_hop.append(
                f"Step {step.step_index + 1} ({step.tool_name}): documents remaining to retrieve before this call "
                f"= {step.gold_titles_remaining_before}; after this call = {step.gold_titles_remaining_after}"
            )

    thread_content_parts = [
        f"Question: {record.question}",
        f"Gold answer: {record.answer}",
        f"Gold supporting titles: {list(record.supporting_titles)}",
    ]
    if missing_gold_titles:
        thread_content_parts.append(f"Missing gold titles after retrieval: {missing_gold_titles}")
    if spurious_titles:
        thread_content_parts.append(f"Spurious retrieved titles: {spurious_titles}")
    thread_content = "\n".join(thread_content_parts)

    extraction_reasoning = [
        f"Model answer: {output.answer!r}",
        f"Gold answer: {record.answer!r}",
    ]

    # Per-component agent feedback the rilixai adapter pulls into the
    # reflective dataset's ``Feedback`` field. ``summarize_prompt``
    # aggregates over the agent's variable number of summarize calls;
    # ``policy_prompt`` analyzes the full tool-call sequence.
    per_component_feedback = build_agent_per_component_feedback(record=record, output=output)

    return {
        "tool_counts": {f"hotpotqa_{name}": count for name, count in tool_counts.items()},
        "tool_calls_detail": tool_calls_detail,
        "retrieval_span_candidates": retrieval_span_candidates,
        "thread_content": thread_content,
        "trace_evidence": {
            "retrieval_reasoning": retrieval_reasoning,
            "extraction_reasoning": extraction_reasoning,
            "documents_remaining_per_hop": documents_remaining_per_hop,
            "policy_reasoning": policy_reasoning,
            "per_component_feedback": per_component_feedback,
        },
        "hotpotqa": {
            "mode": agent_kind + "_agent",
            "retrieval_mode": config.retrieval_mode,
            "missing_gold_titles": missing_gold_titles,
            "spurious_titles": spurious_titles,
            "num_total_steps": len(output.tool_calls),
            "tool_counts": tool_counts,
        },
    }


def _tool_call_for_agent_step(step: AgentToolCall) -> dict[str, Any]:
    return {
        "tool": f"hotpotqa_{step.tool_name}" if step.tool_name else "hotpotqa_unknown",
        "step_index": step.step_index,
        "args": dict(step.tool_args),
        "return": {
            "observation": step.observation,
            "gold_titles_remaining_before": list(step.gold_titles_remaining_before),
            "gold_titles_remaining_after": list(step.gold_titles_remaining_after),
        },
        "thought": step.thought,
    }


# ─── Shared helpers ─────────────────────────────────────────────────────


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


def _bare_openai_model(pydantic_spec: str) -> str:
    """Strip the provider prefix from a PydanticAI model spec.

    PydanticAI uses ``"openai:gpt-4.1-mini"``; the raw OpenAI
    ``chat.completions.create`` call inside the summarize tool wants
    ``"gpt-4.1-mini"`` (no provider prefix). Returns the original
    string when no ``:`` is present so non-openai PydanticAI specs
    that already lack a prefix still pass through unchanged.
    """
    _, separator, model = pydantic_spec.partition(":")
    return model if separator else pydantic_spec
