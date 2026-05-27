"""Per-component textual feedback for the PydanticAI agent variant.

The agent has two optimizable components — ``policy_prompt`` (its
tool-use system prompt) and ``summarize_prompt`` (the summarize
sub-agent's system prompt, reused across every summarize call). Both
get rich, gold-supervised textual feedback the reflection LM reads when
rewriting them.

Decoupled from the workflow's structurally-different per-module
feedback because the agent's tool sequence is variable: the agent
decides whether to summarize zero times, once, twice, or more. The
feedback functions here aggregate across whatever the agent actually
did.
"""

from __future__ import annotations

from ..dataset import HotpotQARecord
from ..gold_helpers import ideal_summary_from_supporting_facts
from ..hotpot_eval import exact_match_score
from .types import AgentToolCall, HotpotQAAgentOutput


# Component names the agent attaches to the rilixai PromptCandidate.
# Defined here so the runtime and the agent's tests can import them
# without pulling in PydanticAI.
AGENT_POLICY_COMPONENT = "policy_prompt"
AGENT_SUMMARIZE_COMPONENT = "summarize_prompt"


def build_agent_per_component_feedback(
    *,
    record: HotpotQARecord,
    output: HotpotQAAgentOutput,
) -> dict[str, str]:
    """Return per-component feedback strings the rilixai adapter consumes.

    Keys map onto the agent's two optimizable components; values are
    the strings the reflection LM sees in the reflective dataset's
    ``Feedback`` field when rewriting that component.
    """
    return {
        AGENT_SUMMARIZE_COMPONENT: _summarize_feedback(record=record, output=output),
        AGENT_POLICY_COMPONENT: _policy_prompt_feedback(record=record, output=output),
    }


# ─── summarize_prompt feedback ─────────────────────────────────────────


def _summarize_feedback(
    *,
    record: HotpotQARecord,
    output: HotpotQAAgentOutput,
) -> str:
    """Feedback for the agent's summarize sub-agent prompt.

    The agent's summarize tool may be called zero or more times per
    case. We surface every call's input shape (question, optional prior
    context, number of passages) and produced summary alongside the
    gold supporting facts and an ideal summary string built from those
    facts. The reflection LM then has concrete material to rewrite the
    summarize prompt against — the same paper-style structure the
    workflow uses, but aggregated over the agent's variable call count.
    """
    summarize_calls = [step for step in output.tool_calls if step.tool_name == "summarize"]
    ideal_summary = ideal_summary_from_supporting_facts(record)
    gold_titles = list(record.supporting_titles)

    if not summarize_calls:
        return (
            f"""You are optimizing the **summarize prompt** of a multi-hop QA agent.

**Question:** "{record.question}"
**Correct answer:** "{record.answer}"
**Gold supporting titles:** {gold_titles}

**Analysis:** The agent did not call `summarize` on this case. If """
            f"""summarizing retrieved evidence would have helped the agent reach the correct answer — """
            f"""for instance by pivoting through entities in hop-1 evidence — your prompt may not be """
            f"""making its value clear to the agent. Consider rewriting the summarize prompt so the """
            f"""agent learns when to invoke summarization and what shape a useful summary takes.

**An ideal summary covering the gold supporting facts for this question would have included:**
   {ideal_summary}"""
        )

    call_blocks = [_format_summarize_call(idx, step) for idx, step in enumerate(summarize_calls, start=1)]
    correctness = "correct" if exact_match_score(output.answer, record.answer) else "incorrect"

    return f"""You are optimizing the **summarize prompt** of a multi-hop QA agent. The prompt is reused for every call to the agent's `summarize(question, passages, context=None)` tool.

**Question:** "{record.question}"
**Correct answer:** "{record.answer}"
**Agent's answer:** "{output.answer}" ({correctness})
**Gold supporting titles:** {gold_titles}

**summarize tool calls on this case ({len(summarize_calls)} total):**
{chr(10).join(call_blocks)}

**An ideal summary covering the gold supporting facts for this question would have included:**
   {ideal_summary}

**Feedback for improvement:**
- Are the summaries dense with the entities and bridging facts the agent will need for downstream retrievals and the final answer?
- When `context` is provided, does the summary integrate it rather than restating it?
- Does the summarize prompt make clear that the summary will be consumed by the agent itself (not the user) — so brevity and information density beat polish?
- If a summarize call missed a key supporting fact present in its passages, your prompt should add explicit guidance to cover those entity / relation classes."""


def _format_summarize_call(call_index: int, step: AgentToolCall) -> str:
    # ``step.tool_args`` is harvested from PydanticAI's message stream,
    # which serializes tool arguments under the actual parameter names.
    # The agent's summarize signature is
    # ``summarize(question, passages, context=None)``, so we read
    # ``passages`` (a list) and the optional ``context``.
    # Earlier versions of this function read ``num_passages`` /
    # ``has_context`` — those are keys present in the agent's internal
    # ``deps.tool_invocations`` log, NOT in ``step.tool_args``. The
    # mismatch made every summarize call render as
    # ``num_passages=?, has_context=False``, which silently misled
    # the reflection LM about the call structure.
    args = step.tool_args
    passages_value = args.get("passages")
    num_passages: int | str = len(passages_value) if isinstance(passages_value, list) else "?"
    has_context = args.get("context") is not None
    obs_preview = step.observation if len(step.observation) <= 600 else step.observation[:600] + "…"
    return (
        f"  Call {call_index}: num_passages={num_passages}, has_context={has_context}\n"
        f"    produced summary: {obs_preview!r}\n"
        f"    gold titles still missing after this call: {step.gold_titles_remaining_after}"
    )


# ─── policy_prompt feedback ─────────────────────────────────────────────


def _policy_prompt_feedback(
    *,
    record: HotpotQARecord,
    output: HotpotQAAgentOutput,
) -> str:
    """Feedback for the agent's tool-flow policy prompt.

    Covers the full tool-call sequence (retrieve_k, summarize,
    structured-answer terminator). Includes an "ideal trajectory"
    sketch derived from the gold supporting facts so the reflection LM
    has a concrete reference shape this case should have produced.
    """
    gold_titles = {t.strip().lower() for t in record.supporting_titles}
    retrieved_titles = {p.title.strip().lower() for p in output.retrieved_paragraphs}
    relevant_retrieved = sorted(
        title for title in record.supporting_titles if title.strip().lower() in retrieved_titles
    )
    missing_titles = sorted(
        title for title in record.supporting_titles if title.strip().lower() not in retrieved_titles
    )
    spurious_titles = sorted(
        p.title for p in output.retrieved_paragraphs if p.title.strip().lower() not in gold_titles
    )
    title_to_sentences = {p.title: p.sentences for p in record.paragraphs}
    missing_full_text = [f"{title} | {''.join(title_to_sentences.get(title, ()))}" for title in missing_titles]

    trajectory_lines = _format_trajectory(output.tool_calls)
    ideal_trajectory = _build_ideal_trajectory(record)
    correctness = "correct" if exact_match_score(output.answer, record.answer) else "incorrect"

    return f"""You are optimizing the **tool-use policy** of a multi-hop QA agent. The agent has tools `retrieve_k(query)` and `summarize(question, passages, context=None)`, and produces a structured `HotpotQAOutput(answer)` to terminate.

**Question:** "{record.question}"
**Correct answer:** "{record.answer}"
**Agent's answer:** "{output.answer}" ({correctness})

**Agent trajectory (in order):**
{trajectory_lines}

**Retrieval analysis:**
- Gold supporting titles needed: {list(record.supporting_titles)}
- Successfully retrieved gold titles: {relevant_retrieved}
- Missing gold titles after the trajectory: {missing_titles}
- Spurious (off-target) retrievals: {spurious_titles}

**Missed-evidence detail:**
{_render_missed_evidence(missing_full_text)}

**Ideal trajectory shape for this question:**
{ideal_trajectory}

**Feedback for improvement:**
- Did the agent retrieve the bridging entity first, then pivot to retrieve the second-hop evidence? Bridge questions usually need a pivot through an entity surfaced in the first retrieval.
- For comparison questions, did the agent issue an independent retrieval per compared entity?
- Did the agent call `summarize` at points where compressing retrieved evidence would help the next decision — e.g. after retrieving hop-1 docs before formulating the hop-2 query?
- Was the structured `HotpotQAOutput` produced as soon as enough evidence was available, or did the agent over-retrieve / under-retrieve?
- Rewrite the policy prompt so the agent learns the pattern that would have produced the ideal trajectory for this question shape."""


def _format_trajectory(tool_calls: list[AgentToolCall]) -> str:
    if not tool_calls:
        return "  (no tool calls)"
    lines: list[str] = []
    for step in tool_calls:
        args_repr = ", ".join(f"{k}={v!r}" for k, v in step.tool_args.items() if k not in {"passages", "context"})
        obs_preview = (step.observation[:120] + "…") if len(step.observation) > 120 else step.observation
        thought_preview = (step.thought[:120] + "…") if len(step.thought) > 120 else step.thought
        lines.append(
            f"  Step {step.step_index + 1}: {step.tool_name}({args_repr})\n"
            f"    thought: {thought_preview!r}\n"
            f"    obs: {obs_preview!r}"
        )
    return "\n".join(lines)


def _build_ideal_trajectory(record: HotpotQARecord) -> str:
    """A "what the agent should have done" sketch from this case's gold facts."""
    titles = list(record.supporting_titles[:2]) or ["<first gold title>", "<second gold title>"]
    title_to_sentences = {p.title: p.sentences for p in record.paragraphs}
    cues: list[str] = []
    for title in titles:
        sentences = title_to_sentences.get(title, ())
        cue = sentences[0] if sentences else title
        cues.append(f"{title}: {cue}")
    pivots = [titles[0], titles[1] if len(titles) > 1 else titles[0]]
    return (
        f"  Step 1: retrieve_k(query='{pivots[0]}')                      # bridge/comparison hop-1\n"
        f"  Step 2: summarize(question, passages)                         # compress hop-1 evidence\n"
        f"  Step 3: retrieve_k(query='{pivots[1]}')                      # pivot using hop-1 summary\n"
        f"  Step 4: summarize(question, passages, context=<hop-1 summary>) # integrate both hops\n"
        f"  Step 5: produce HotpotQAOutput(answer='{record.answer}')      # structured terminator\n\n"
        f"  Evidence cues from gold supporting facts:\n   " + "\n   ".join(cues)
    )


def _render_missed_evidence(missing_full_text: list[str]) -> str:
    if not missing_full_text:
        return "  (all gold supporting titles were retrieved)"
    return "\n".join(f"  - {block}" for block in missing_full_text)


__all__ = [
    "AGENT_POLICY_COMPONENT",
    "AGENT_SUMMARIZE_COMPONENT",
    "build_agent_per_component_feedback",
]
