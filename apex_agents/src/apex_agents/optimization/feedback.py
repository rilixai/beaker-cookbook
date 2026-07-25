"""Per-component textual feedback for the APEX-Agents agent.

Three optimizable components, each given rich textual feedback the
reflection LM reads when rewriting it:

* ``system_prompt`` — feedback on the agent's *trajectory*: did it
  plan with todos, discover + add the right domain tools, thrash, or
  hit the step limit?
* ``task_template`` — feedback on the *task framing* and the final
  answer's coverage of the rubric. Includes a ``{{task}}``-preservation
  check (mirrors SWE-bench's preservation guard) so GEPA can't drop
  the substitution variable.
* ``resum_summary_prompt`` — feedback on context compaction: did ReSum
  fire, and did the agent still finish coherently afterwards?
"""

from __future__ import annotations

from collections.abc import Sequence

from ..agent.prompts import (
    RESUM_SUMMARY_PROMPT_COMPONENT,
    SYSTEM_PROMPT_COMPONENT,
    TASK_TEMPLATE_COMPONENT,
)
from ..agent.types import AgentToolCall, ApexAgentsAgentOutput
from ..data.dataset import ApexAgentsRecord


def build_apex_per_component_feedback(
    *,
    record: ApexAgentsRecord,
    output: ApexAgentsAgentOutput,
) -> dict[str, str]:
    """Return per-component feedback strings the rilixai adapter consumes."""
    return {
        SYSTEM_PROMPT_COMPONENT: _system_prompt_feedback(record=record, output=output),
        TASK_TEMPLATE_COMPONENT: _task_template_feedback(record=record, output=output),
        RESUM_SUMMARY_PROMPT_COMPONENT: _resum_summary_prompt_feedback(record=record, output=output),
    }


# Tools that actually read the workspace. If NONE of these were called
# the agent answered "blind" (from parametric knowledge) or punted — the
# dominant score-loss mode on Law held-out (~54% of cases). This must be
# the loud headline of the system_prompt feedback so GEPA's reflection
# LM rewrites the prompt to force exploration, not a buried bullet.
_FILE_TOOLS = frozenset({"list_files", "read_file", "read_pdf", "read_spreadsheet", "read_docx", "search_files"})

# Punt phrases an agent uses when it gives up instead of reading files.
_PUNT_MARKERS = (
    "i cannot",
    "i am unable",
    "i was unable",
    "unable to",
    "please provide",
    "not provided",
    "without the",
    "do not have access",
    "cannot proceed",
    "no information",
)


def _tool_usage(messages: Sequence[AgentToolCall]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for m in messages:
        if m.role == "assistant" and m.tool_name:
            counts[m.tool_name] = counts.get(m.tool_name, 0) + 1
    return counts


def _non_exploration_headline(usage: dict[str, int], output: ApexAgentsAgentOutput) -> str:
    """Lead line for the system_prompt feedback flagging blind-answering / punting."""
    explored = any(t in usage for t in _FILE_TOOLS)
    answer_lc = (output.final_answer or "").lower()
    punted = output.status == "blocked" or any(m in answer_lc for m in _PUNT_MARKERS)
    if not explored:
        return (
            "CRITICAL FAILURE — the agent made ZERO workspace-reading tool calls "
            "(list_files / read_file / read_pdf / read_docx / read_spreadsheet / "
            "search_files) and answered from prior knowledge or gave up. The answer "
            "is in the workspace files; an answer written without reading them scores "
            "~0. The system_prompt MUST force the sequence: list_files → add the "
            "needed file tools → READ the relevant documents → only then answer; and "
            "MUST forbid answering from general knowledge or asking the user for "
            "documents that are in the workspace. This is the single highest-priority "
            "rewrite.\n\n"
        )
    if punted:
        return (
            "HIGH-PRIORITY FAILURE — the agent explored but then PUNTED (status "
            "'blocked' or an 'I cannot / please provide' answer) instead of "
            "committing to a best-effort answer grounded in what it read. The "
            "system_prompt MUST forbid punting: require a concrete, evidence-cited "
            "answer from the documents even under uncertainty, and never request "
            "data that should be in the workspace.\n\n"
        )
    return ""


def _format_trajectory(messages: Sequence[AgentToolCall], *, last_n: int = 8) -> str:
    if not messages:
        return "  (no steps recorded)"
    lines: list[str] = []
    for step in list(messages)[-last_n:]:
        if step.role == "assistant":
            tool_repr = f"tool={step.tool_name!r}" if step.tool_name else "no tool call"
            preview = step.content[:120] + ("…" if len(step.content) > 120 else "")
            lines.append(f"  step {step.step_index}: assistant — {tool_repr}; {preview!r}")
        elif step.role == "tool":
            out_preview = (step.output or "")[:120] + ("…" if len(step.output or "") > 120 else "")
            lines.append(f"  step {step.step_index}: tool {step.tool_name!r} — {out_preview!r}")
        else:
            lines.append(f"  step {step.step_index}: {step.role} — {step.content[:80]!r}")
    return "\n".join(lines)


def _rubric_lines(record: ApexAgentsRecord) -> str:
    if not record.rubric:
        return "  (no rubric criteria)"
    return "\n".join(f"  {i + 1}. ({c.verifier_id}) {c.criteria}" for i, c in enumerate(record.rubric))


def _system_prompt_feedback(*, record: ApexAgentsRecord, output: ApexAgentsAgentOutput) -> str:
    usage = _tool_usage(output.messages)
    used_todo = usage.get("todo_write", 0) > 0
    discovered = usage.get("toolbelt_list_tools", 0) > 0
    added_tools = usage.get("toolbelt_add_tool", 0) > 0
    trajectory = _format_trajectory(output.messages)
    headline = _non_exploration_headline(usage, output)
    return f"""{headline}You are optimizing the **system_prompt** of an APEX-Agents ReAct toolbelt agent. The system_prompt is the model's system message — it establishes the think-before-acting policy, the tool catalogue, the plan/discover/execute/complete workflow, and the rules.

**Task:** {record.task_id} — {record.task_name!r} (world={record.world_id})
**Outcome:** status={output.status!r}, steps={output.total_steps}, cost=${output.total_cost:.4f}, resum_fired={output.resum_count}
**Final answer:** {len(output.final_answer)} chars{", empty" if not output.final_answer else ""}

**Trajectory analysis:**
- Tool usage: {usage or "(no tool calls)"}
- Planned with todo_write: {used_todo}
- Discovered tools via toolbelt_list_tools: {discovered}
- Added domain tools to the toolbelt: {added_tools}
- Hit step limit: {output.status == "max_steps"}

**Rubric the answer is graded against:**
{_rubric_lines(record)}

**Last few trajectory steps:**
{trajectory}

**Feedback for improvement:**
- If the agent never planned with `todo_write`, the system_prompt should make the plan-first workflow more forcing.
- If it never discovered/added domain tools, reinforce the discover→add→execute sequence (the toolbelt starts EMPTY).
- If it hit the step limit, encourage tighter exploration and earlier `final_answer`.
- If it produced an empty final answer, re-emphasize the completion protocol (`final_answer` is rejected with open todos).
- The system_prompt controls *how* the agent uses tools — rewrite it to address the failure mode visible in the trajectory."""


def _task_template_feedback(*, record: ApexAgentsRecord, output: ApexAgentsAgentOutput) -> str:
    answer_preview = output.final_answer[:600] if output.final_answer else "(empty)"
    return f"""You are optimizing the **task_template** of an APEX-Agents agent. The task_template is the first user message; the Jinja2 variable ``{{{{task}}}}`` is substituted with the raw task prompt.

CRITICAL: the rewritten task_template MUST keep the ``{{{{task}}}}`` substitution variable. If it is dropped the agent never sees the task and every rubric criterion fails. Preserve ``{{{{task}}}}`` verbatim.

**Task:** {record.task_id} — {record.task_name!r} (world={record.world_id})
**Task prompt preview:** {record.prompt[:400]!r}
**Outcome:** status={output.status!r}

**Rubric the answer is graded against:**
{_rubric_lines(record)}

**Agent's final answer (first 600 chars):**
{answer_preview}

**Feedback for improvement:**
- If the final answer ignored rubric criteria, the task_template should frame the deliverable so every criterion is addressed (the first rubric entry is the primary objective).
- If the answer was unstructured, the task_template can request a structure that maps onto the rubric.
- Keep ``{{{{task}}}}`` — never inline the task text or drop the variable.
- Rewrite the task_template to lift the answer's rubric coverage without removing ``{{{{task}}}}``."""


def _resum_summary_prompt_feedback(*, record: ApexAgentsRecord, output: ApexAgentsAgentOutput) -> str:
    fired = output.resum_count > 0
    return f"""You are optimizing the **resum_summary_prompt** of an APEX-Agents agent. This prompt compacts a long work session into a reasoning state when the context grows past the ReSum trigger. It MUST keep a literal ``{{conversation}}`` placeholder — it is ``str.format``-substituted with the prior conversation.

**Task:** {record.task_id} (world={record.world_id})
**Outcome:** status={output.status!r}, steps={output.total_steps}, resum_fired={output.resum_count}

**Feedback for improvement:**
- ReSum {"fired " + str(output.resum_count) + " time(s)" if fired else "did not fire"} this run.
- If the agent lost track of file paths / values / todos after a compaction (visible as repeated re-discovery later in the trajectory), the resum_summary_prompt should demand those concrete details be carried forward.
- If the agent finished coherently, keep the prompt's structure but tighten it so the carried state is smaller.
- ALWAYS keep the ``{{conversation}}`` placeholder; without it the summary has nothing to compress."""


def task_template_preserves_task_var(task_template: str) -> bool:
    """Return ``True`` iff the task_template still carries the ``{{task}}`` var.

    Mirrors SWE-bench's ``{{task}}`` preservation check — the feedback
    loop / tests use this so GEPA can't silently drop the substitution
    variable.
    """
    return "{{task}}" in task_template or "{{ task }}" in task_template


__all__ = [
    "RESUM_SUMMARY_PROMPT_COMPONENT",
    "SYSTEM_PROMPT_COMPONENT",
    "TASK_TEMPLATE_COMPONENT",
    "build_apex_per_component_feedback",
    "task_template_preserves_task_var",
]
