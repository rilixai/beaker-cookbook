"""Async ``ExtractionRuntime`` builder for the PydanticAI agent variant.

Wraps :class:`HotpotQAPydanticAgent` as an async runtime the rilixai
adapter can drive. Translates the agent's variable-length tool-call
trajectory into the same trajectory-dict schema the workflow runtime
emits, and populates ``trace_evidence.per_component_feedback`` with
agent-specific feedback strings (the workflow's feedback functions
don't fit because the agent's tool sequence isn't fixed-hop).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from ..dataset import HotpotQARecord
from .feedback import build_agent_per_component_feedback
from .types import AgentToolCall, HotpotQAAgentOutput


if TYPE_CHECKING:
    from ..pipeline import HotpotQAPipelineConfig, HotpotQARunResult


logger = logging.getLogger(__name__)


def build_pydantic_agent_runtime(
    *,
    cfg: HotpotQAPipelineConfig,
    agent: Any | None,
) -> Callable[..., Awaitable[HotpotQARunResult]]:
    """Return an async runtime callable that executes one agent case."""
    # Defer the import so callers using only workflow mode don't need
    # pydantic-ai installed at import time.
    from ..pipeline import HotpotQARunResult, _unpack_runtime_inputs
    from .agent import HotpotQAPydanticAgent

    resolved_agent: HotpotQAPydanticAgent
    if isinstance(agent, HotpotQAPydanticAgent):
        resolved_agent = agent
    elif agent is None:
        if cfg.pydantic_agent_model is None:
            raise ValueError(
                "pydantic_agent mode requires either a pre-built agent or HotpotQAPipelineConfig.pydantic_agent_model."
            )
        resolved_agent = HotpotQAPydanticAgent(
            model=cfg.pydantic_agent_model,
            top_k=cfg.retrieve_k,
            max_iters=cfg.max_iters,
            temperature=cfg.pydantic_agent_temperature,
        )
    else:
        raise TypeError(f"pydantic_agent must be a HotpotQAPydanticAgent or None, got {type(agent).__name__}.")

    # Reuse the workflow's per-case retrieve fn so the agent sees the
    # same retrieval corpus the workflow would for the same case +
    # ``cfg.retrieval_mode``. Before this, ``_do_retrieve`` always
    # searched the case's 10 local paragraphs regardless of mode —
    # making nominally-fullwiki agent runs silently use distractor
    # retrieval and producing non-comparable numbers vs the workflow's
    # fullwiki baseline.
    from ..retrieval import build_retrieve_k_fn_for_case

    async def _runtime(**kwargs: Any) -> HotpotQARunResult:
        record, candidate = _unpack_runtime_inputs(kwargs)
        resolved_agent.apply_candidate(candidate.components)
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
    config: HotpotQAPipelineConfig,
) -> dict[str, Any]:
    """Translate an agent's tool-call trace into rilixai trajectory metadata."""
    from ..pipeline import _truncate

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
