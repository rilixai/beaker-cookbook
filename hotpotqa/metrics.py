"""Trajectory metadata builder for the HotpotQA agent.

Translates an agent's tool-call trace into the ``run_metrics`` dict the
optimizer reads. The runner's ``_package_result`` (in
:mod:`hotpotqa.rilixai_spec`) is the only caller; the paper-faithful answer
scoring lives next to the ``@spec`` runner as :class:`HotpotQAMetrics`.
"""

from __future__ import annotations

from typing import Any

from .agent.types import AgentToolCall, HotpotQAAgentOutput
from .config import HotpotQAConfig
from .data.dataset import HotpotQARecord


def build_agent_run_metrics(
    *,
    record: HotpotQARecord,
    output: HotpotQAAgentOutput,
    agent_kind: str,
    config: HotpotQAConfig,
) -> dict[str, Any]:
    """Translate an agent's tool-call trace into rilixai trajectory metadata.

    Owns only the domain-specific trace evidence (per-hop retrieval reasoning,
    documents-remaining, missing/spurious titles). Per-component feedback flows
    separately through ``@spec(feedback=HotpotQAFeedback)``; the runner merges
    it into ``trace_evidence.per_component_feedback`` in ``_package_result``.
    """
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
        },
        "hotpotqa": {
            "mode": agent_kind + "_agent",
            "retrieval_mode": config.retrieval_mode,
            "missing_gold_titles": missing_gold_titles,
            "spurious_titles": spurious_titles,
            "num_total_steps": len(output.tool_calls),
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


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"
