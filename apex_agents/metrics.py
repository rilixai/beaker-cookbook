"""The LLM rubric judge for APEX-Agents scoring.

A single optimization field, ``rubric_pass_rate``, scores the agent's
final answer against the task's rubric. Each rubric criterion is
graded binary Met / Not-met by an LLM judge (Mercor's Archipelago
default is ``gemini/gemini-2.5-flash`` — the "output_llm" verifier).
The task score is the fraction of criteria Met.

The runner owns the judge call (it has the agent's final answer + the
task prompt + the rubric) and stashes ``rubric_pass_rate`` on its
result; the ``ApexAgentsMetrics`` calculator in
:mod:`apex_agents.rilixai_spec` reads that precomputed float via a
custom comparator.

:func:`build_rubric_judge` returns the callable the runner uses; its
``llm`` is injectable so tests stub the verdicts and zero network
fires.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from rilixai.prompt_optimization.protocols import ErrorOutput

from .agent.types import ApexAgentsAgentOutput
from .config import ApexAgentsConfig
from .data.dataset import ApexAgentsRecord


logger = logging.getLogger(__name__)


__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "RUBRIC_FIELD",
    "RubricJudge",
    "build_apex_agents_run_metrics",
    "build_rubric_judge",
    "coerce_pass_rate",
    "score_rubric",
]


RUBRIC_FIELD = "rubric_pass_rate"

DEFAULT_JUDGE_MODEL = "gemini/gemini-2.5-flash"
# Bound each judge call so a hung provider request fails the criterion
# (conservatively Not-met) fast instead of stalling the whole run.
DEFAULT_JUDGE_TIMEOUT_S = 120.0
DEFAULT_JUDGE_NUM_RETRIES = 2


# Callable contract: ``judge(criterion, answer, task_prompt) -> bool``.
RubricJudge = Callable[[str, str, str], bool]


def coerce_pass_rate(value: Any) -> float:
    """Coerce a ``rubric_pass_rate`` field value into a [0, 1] float.

    The runner stashes the judge-computed float on its result; the
    ``ApexAgentsMetrics`` comparator calls this to read it back robustly.
    """
    if isinstance(value, ErrorOutput):
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        score = float(value)
        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return score
    if isinstance(value, str):
        try:
            return coerce_pass_rate(float(value))
        except ValueError:
            return 0.0
    return 0.0


# Primary signal: an explicit, labeled verdict line the judge is asked
# to emit *after* any reasoning. Anchoring on a label makes the parse
# immune to verbose / "thinking" judges (e.g. gemini-2.5-flash) whose
# free-text reasoning otherwise trips a bare-substring scan ("...there
# are no errors" → matched "no" → wrongly Not-met). We take the LAST
# tag so a verdict stated after the reasoning wins.
_VERDICT_TAG_RE = re.compile(r"VERDICT\s*[:\-]?\s*(NOT[\s_-]?MET|MET)\b", re.IGNORECASE)


def _normalize_last_line(text: str) -> str:
    for line in reversed(text.strip().splitlines()):
        s = line.strip().strip("*_`#.\"' \t").upper()
        if s:
            return s
    return ""


def _parse_verdict(text: str) -> bool:
    """Parse an LLM judge reply into a Met (True) / Not-met (False) boolean.

    Robust to reasoning-model verbosity:

    1. Prefer the explicit ``VERDICT: MET`` / ``VERDICT: NOT MET`` tag
       the judge prompt asks for; the LAST occurrence wins (the verdict
       stated after any reasoning).
    2. Fallback: inspect ONLY the last non-empty line — not the whole
       blob — so earlier reasoning prose can't poison the parse.
    3. Anything still ambiguous → conservative Not-met, logged at
       WARNING so a systematically-unparseable judge is visible (this
       previously hid behind a bare-substring scan and silently
       depressed scores for verbose judges).
    """
    tags = _VERDICT_TAG_RE.findall(text or "")
    if tags:
        return "NOT" not in tags[-1].upper()

    last = _normalize_last_line(text or "")
    # Match MET / NOT MET as a leading *token* (``\b``) so a last line that
    # merely starts with "MET" — METHOD, METADATA, METRICS — isn't mis-scored
    # as Met (which would silently inflate rubric_pass_rate). Mirrors the
    # ``NOT[\s_-]?MET`` shape of _VERDICT_TAG_RE; ``last`` is already upper-cased.
    if re.match(r"NOT[\s_-]?MET\b", last) or last in {"NO", "FALSE", "FAIL", "FAILED"}:
        return False
    if re.match(r"MET\b", last) or last in {"YES", "TRUE", "PASS", "PASSED", "SATISFIED"}:
        return True

    logger.warning(
        "Rubric judge reply had no parseable verdict (scoring Not-met): %r",
        (text or "")[:200],
    )
    return False


def build_rubric_judge(
    model: str = DEFAULT_JUDGE_MODEL,
    llm: Callable[..., Any] | None = None,
    *,
    timeout: float = DEFAULT_JUDGE_TIMEOUT_S,
    num_retries: int = DEFAULT_JUDGE_NUM_RETRIES,
) -> RubricJudge:
    """Return a ``judge(criterion, answer, task_prompt) -> bool`` callable.

    ``llm`` is injectable for tests: pass a stub
    ``llm(model=..., messages=...) -> str | response`` returning a
    scripted verdict so zero network fires. In production ``llm`` is
    ``None`` and litellm is imported lazily. ``timeout``/``num_retries``
    bound each judge call so a hung provider request fails that
    criterion (→ Not-met) fast instead of stalling the whole run.
    """

    def _call_llm(messages: list[dict[str, str]]) -> str:
        if llm is not None:
            result = llm(model=model, messages=messages)
            # Accept either a bare string verdict or a litellm-shaped
            # response object/dict (tests usually return a string).
            if isinstance(result, str):
                return result
            if isinstance(result, Mapping):
                choices = result.get("choices")
                if isinstance(choices, Sequence) and choices:
                    msg = choices[0].get("message") if isinstance(choices[0], Mapping) else None
                    if isinstance(msg, Mapping):
                        return str(msg.get("content") or "")
                return str(result.get("content") or "")
            return str(result)
        import litellm  # lazy import — keeps the module offline-importable

        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=0.0,
            timeout=timeout,
            num_retries=num_retries,
        )
        return str(response.choices[0].message.content or "")

    def _judge(criterion: str, answer: str, task_prompt: str) -> bool:
        prompt = (
            "You are grading whether an AI agent's answer satisfies a single rubric "
            "criterion for a professional knowledge-work task.\n\n"
            f"## Task\n{task_prompt}\n\n"
            f"## Rubric criterion\n{criterion}\n\n"
            f"## Agent's final answer\n{answer}\n\n"
            "Decide whether the agent's answer satisfies this single criterion. "
            "You may reason briefly first. Then end your reply with the verdict on "
            "its own final line in EXACTLY this form:\n"
            "VERDICT: MET\n"
            "or\n"
            "VERDICT: NOT MET"
        )
        try:
            verdict = _call_llm([{"role": "user", "content": prompt}])
        except Exception as exc:  # pragma: no cover - judge outage is a miss
            logger.warning("Rubric judge call failed (scoring Not-met): %s", exc)
            return False
        return _parse_verdict(verdict)

    return _judge


def score_rubric(
    *,
    rubric: Sequence[Mapping[str, Any]],
    answer: str,
    task_prompt: str,
    judge: RubricJudge,
) -> float:
    """Return the fraction of rubric criteria the judge marked Met.

    An empty rubric yields ``0.0`` (an unscoreable task). The runtime
    calls this once per case and stashes the float on the result.
    """
    criteria = [str(c.get("criteria") or "") for c in rubric if str(c.get("criteria") or "").strip()]
    if not criteria:
        return 0.0
    met = 0
    for criterion in criteria:
        try:
            if judge(criterion, answer, task_prompt):
                met += 1
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Rubric judge raised on a criterion: %s", exc)
    return met / len(criteria)


# ─── Trajectory metadata builder (called by the runner's _package_result) ─


def build_apex_agents_run_metrics(
    *,
    record: ApexAgentsRecord,
    output: ApexAgentsAgentOutput,
    config: ApexAgentsConfig,
    rubric_pass_rate: float,
) -> dict[str, Any]:
    """Translate the agent's output into rilixai trajectory metadata.

    Per-component feedback is no longer embedded here — the runner asks
    rilixai for either default GenericFeedback or the optional
    ``@spec(feedback=...)`` class, then merges it into
    ``trace_evidence.per_component_feedback``. This builder owns only the
    domain-specific trace evidence.
    """
    policy_reasoning = [
        _truncate(m.content, config.max_preview_chars) for m in output.messages if m.role == "assistant"
    ][:5]
    tools_called = [m.tool_name for m in output.messages if m.role == "assistant" and m.tool_name]

    tool_counts: dict[str, int] = {}
    for name in tools_called:
        tool_counts[name] = tool_counts.get(name, 0) + 1

    tool_calls_detail: list[dict[str, Any]] = []
    for step in output.messages:
        tool_calls_detail.append(
            {
                "step_index": step.step_index,
                "role": step.role,
                "tool_name": step.tool_name,
                "tool_args": step.tool_args,
                "output_preview": (_truncate(step.output or "", config.max_preview_chars) if step.output else None),
                "content_preview": _truncate(step.content, config.max_preview_chars),
            }
        )

    return {
        "tool_counts": tool_counts,
        "tool_calls_detail": tool_calls_detail,
        "trace_evidence": {
            "policy_reasoning": policy_reasoning,
            "tools_called": tools_called[:25],
        },
        "apex_agents": {
            "task_id": record.task_id,
            "world_id": record.world_id,
            "domain": record.domain,
            "status": output.status,
            "rubric_pass_rate": rubric_pass_rate,
            "num_rubric_criteria": len(record.rubric),
            "total_steps": output.total_steps,
            "total_cost": output.total_cost,
            "wall_seconds": output.wall_seconds,
            "resum_count": output.resum_count,
            "answer_chars": len(output.final_answer),
        },
    }


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"
