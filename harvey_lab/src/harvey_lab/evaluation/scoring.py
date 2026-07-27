"""Rubric scoring for Harvey LAB: batched judge + all-pass aggregation.

Harvey LAB grades each task against an array of equally-weighted binary
criteria. An LLM judge decides PASS / FAIL against each criterion's
``match_criteria`` standard, seeing ONLY the deliverables that criterion
names (deliverable-scoped context — LAB tasks carry ~60 criteria, so scoping
keeps judging focused). The headline benchmark metric is **all-pass**: a
task scores ``1.0`` iff every criterion passes, else ``0.0``.

Rather than one LLM call per criterion (Harvey's reference scorer), criteria
that share the same deliverable scope are graded in **batches** of
``batch_size`` per call — batched verification is ~an order of magnitude
cheaper at LAB's scale, at near-frontier agreement:

* https://www.langchain.com/blog/designing-efficient-verifiers-for-legal-agents
  (batch the rubric into one call; cheaper open models like DeepSeek v4 Flash)
* https://www.appliedcompute.com/case-studies/harvey (GPT-5 Mini, 4/call)

Two numbers come out per task:

* ``all_pass`` (0/1) — the share of tasks where every criterion passes.
* ``criterion_pass_rate`` (continuous) — the fraction of criteria that
  passed; LAB-AA's default headline metric, and a denser view of the same
  grading (all-pass is very sparse on a ~60-criterion task: one missed
  criterion zeroes the whole task).

Two LAB-AA grading rules are enforced here:

* Deliverable filenames match **exactly** — a near-miss filename counts as not
  produced.
* A criterion fails outright, without reaching the judge, only when *none* of
  its declared deliverables exist. A partial submission is still judged, with
  the absent files marked as such.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any


logger = logging.getLogger(__name__)


ALL_PASS_FIELD = "all_pass"
ALL_PASS_RATE_FIELD = "all_pass_rate"
CRITERION_PASS_RATE_FIELD = "criterion_pass_rate"

DEFAULT_JUDGE_MODEL = "openrouter/deepseek/deepseek-v4-flash"
DEFAULT_JUDGE_BATCH_SIZE = 8
DEFAULT_JUDGE_TIMEOUT_S = 120.0
# LAB-AA retries API failures aggressively rather than letting a transient
# outage silently score a criterion FAIL.
DEFAULT_JUDGE_NUM_RETRIES = 8
DEFAULT_MAX_DELIVERABLE_CHARS = 40_000


# A batched judge: ``judge(task_description, criteria, agent_output) ->
# {criterion_id: passed}`` for the criteria in one same-scope batch.
BatchJudge = Callable[[str, Sequence[Mapping[str, Any]], str], dict[str, bool]]


# Batched adaptation of Harvey's reference judge prompt
# (evaluation/prompts/rubric_criterion.txt): the judge labels every criterion
# in the batch in a single call.
_JUDGE_PROMPT_TEMPLATE = """You are evaluating a legal AI agent's work product against a set of quality criteria.

## Task
{task_description}

## Agent's Output
{agent_output}

## Criteria
{criteria_block}

## Instructions
Evaluate the agent's output against EACH criterion above, independently.
- **pass**: the agent's output satisfies that criterion as described
- **fail**: the agent's output does not satisfy that criterion as described

Respond with JSON only — one object per criterion, echoing its id:

```json
{{
  "verdicts": [
    {{"id": "<criterion id>", "verdict": "pass" | "fail", "reasoning": "Brief explanation"}}
  ]
}}
```"""


def _criteria_block(criteria: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for c in criteria:
        title = str(c.get("title") or "(untitled criterion)")
        parts.append(f"### id: {c.get('id')}\n**{title}**\n\n{c.get('match_criteria') or ''}")
    return "\n\n".join(parts)


def _extract_verdicts_payload(text: str) -> Mapping[str, Any] | None:
    """Recover the ``{"verdicts": [...]}`` object from a judge reply.

    The model may wrap the JSON in markdown fences or add prose before/after
    it (including text with its own braces), so a greedy first-to-last-brace
    match would over-capture and fail to parse. Instead scan for each balanced
    ``{...}`` object (by brace depth, ignoring braces inside strings) and
    return the first that parses AND holds a ``verdicts`` key.
    """
    blob = text.replace("```json", "").replace("```", "")
    for start, obj in enumerate(blob):
        if obj != "{":
            continue
        depth = 0
        in_str = False
        escaped = False
        for end in range(start, len(blob)):
            ch = blob[end]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(blob[start : end + 1])
                    except Exception:
                        break  # not valid JSON from this "{"; try the next one
                    if isinstance(parsed, Mapping) and "verdicts" in parsed:
                        return parsed
                    break
    return None


def _parse_batch_verdicts(text: str, ids: Sequence[str]) -> dict[str, bool]:
    """Parse a batched judge reply into ``{criterion_id: passed}``.

    Reads the ``{"verdicts": [{"id", "verdict"}, ...]}`` shape. Any criterion
    the judge omits or leaves ambiguous → conservative FAIL (a missed verdict
    must not inflate the score), logged so an unparseable judge is visible.
    """
    result: dict[str, bool] = dict.fromkeys(ids, False)
    payload = _extract_verdicts_payload(text or "")
    raw = payload.get("verdicts") if isinstance(payload, Mapping) else None
    # ``verdicts`` may be JSON null or a non-list; only iterate a real sequence.
    entries: Sequence[Any] = raw if isinstance(raw, (list, tuple)) else []
    seen = 0
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        cid = str(entry.get("id") or "")
        if cid not in result:
            continue
        verdict = str(entry.get("verdict") or "").strip().lower()
        result[cid] = verdict == "pass"
        seen += 1
    if seen < len(ids):
        logger.warning("Judge returned %d/%d verdicts; missing ones scored FAIL.", seen, len(ids))
    return result


def build_rubric_judge(
    model: str = DEFAULT_JUDGE_MODEL,
    llm: Callable[..., Any] | None = None,
    *,
    timeout: float = DEFAULT_JUDGE_TIMEOUT_S,
    num_retries: int = DEFAULT_JUDGE_NUM_RETRIES,
) -> BatchJudge:
    """Return a batched ``judge(task, criteria, output) -> {id: passed}``.

    ``llm`` is injectable for tests: pass ``llm(model=..., messages=...) ->
    str | response`` returning a scripted batch verdict so zero network fires.
    In production ``llm`` is ``None`` and litellm is imported lazily.
    """

    def _call_llm(messages: list[dict[str, str]]) -> str:
        if llm is not None:
            result = llm(model=model, messages=messages)
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

    def _judge(task_description: str, criteria: Sequence[Mapping[str, Any]], agent_output: str) -> dict[str, bool]:
        ids = [str(c.get("id")) for c in criteria]
        prompt = _JUDGE_PROMPT_TEMPLATE.format(
            task_description=task_description,
            agent_output=agent_output or "(the agent produced no deliverables)",
            criteria_block=_criteria_block(criteria),
        )
        try:
            reply = _call_llm([{"role": "user", "content": prompt}])
        except Exception as exc:  # pragma: no cover - judge outage is a batch of FAILs
            logger.warning("Rubric judge call failed (scoring FAIL): %s", exc)
            return dict.fromkeys(ids, False)
        return _parse_batch_verdicts(reply, ids)

    return _judge


def _scope_deliverables(
    criterion_deliverables: Sequence[str],
    deliverables: Mapping[str, str],
    *,
    max_chars: int,
) -> str:
    """Concatenate only the deliverables a criterion names (deliverable-scoping).

    Filenames are matched **exactly** — LAB-AA counts a near-miss filename as
    not produced, which is stricter than Harvey's best-effort matching. A file
    the criterion declares but the agent never produced is included as an
    explicit *absent* marker rather than silently dropped, so the judge grades a
    partial submission on what is actually there (AA judges a criterion whenever
    at least one of its files exists, marking the rest absent).

    A criterion that names no deliverable at all falls back to every produced
    deliverable so the judge still has context.
    """
    wanted = [str(d) for d in criterion_deliverables if d]
    if not wanted:
        return "\n\n".join(f"### {name}\n{text[:max_chars]}" for name, text in deliverables.items())
    parts: list[str] = []
    for name in wanted:
        if name in deliverables:
            parts.append(f"### {name}\n{deliverables[name][:max_chars]}")
        else:
            parts.append(f"### {name}\n(this deliverable was not produced)")
    return "\n\n".join(parts)


def _scope_key(criterion: Mapping[str, Any]) -> tuple[str, ...]:
    """The deliverable-scope a criterion is graded under (its batching group)."""
    return tuple(sorted(str(d) for d in (criterion.get("deliverables") or ()) if d))


def score_rubric(
    *,
    criteria: Sequence[Mapping[str, Any]],
    deliverables: Mapping[str, str],
    task_description: str,
    judge: BatchJudge,
    batch_size: int = DEFAULT_JUDGE_BATCH_SIZE,
    max_deliverable_chars: int = DEFAULT_MAX_DELIVERABLE_CHARS,
) -> dict[str, Any]:
    """Grade every criterion and aggregate into the LAB all-pass result.

    Criteria that share a deliverable scope are graded together, ``batch_size``
    per judge call. Returns ``{"all_pass", "criterion_pass_rate", "passed",
    "total_criteria", "verdicts"}``. An empty rubric yields
    ``total_criteria == 0`` (an unscoreable task) — the caller signals that
    upstream so it is excluded from the mean.
    """
    scoreable = [c for c in criteria if str(c.get("match_criteria") or "").strip()]
    total_criteria = len(scoreable)
    if total_criteria == 0:
        return {ALL_PASS_FIELD: 0.0, CRITERION_PASS_RATE_FIELD: 0.0, "passed": 0, "total_criteria": 0, "verdicts": []}

    # Group by deliverable scope so each judge call sees the right documents,
    # then chunk each group into batches.
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for c in scoreable:
        groups.setdefault(_scope_key(c), []).append(c)

    passed_by_id: dict[str, bool] = {}
    for scope, group in groups.items():
        # AA's rule: a criterion fails outright, WITHOUT being shown to the
        # judge, only when none of its declared deliverables were produced. A
        # partial submission is still judged, with the missing files marked
        # absent by ``_scope_deliverables``.
        if scope and not any(name in deliverables for name in scope):
            for c in group:
                passed_by_id[str(c.get("id"))] = False
            continue
        scoped_output = _scope_deliverables(scope, deliverables, max_chars=max_deliverable_chars)
        for start in range(0, len(group), max(1, batch_size)):
            batch = group[start : start + max(1, batch_size)]
            try:
                verdicts = judge(task_description, batch, scoped_output)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Rubric judge raised on a batch: %s", exc)
                verdicts = {str(c.get("id")): False for c in batch}
            for c in batch:
                passed_by_id[str(c.get("id"))] = bool(verdicts.get(str(c.get("id")), False))

    verdicts_out = [{"id": c.get("id"), "passed": passed_by_id.get(str(c.get("id")), False)} for c in scoreable]
    n_passed = sum(1 for v in verdicts_out if v["passed"])
    return {
        ALL_PASS_FIELD: 1.0 if n_passed == total_criteria else 0.0,
        CRITERION_PASS_RATE_FIELD: n_passed / total_criteria,
        "passed": n_passed,
        "total_criteria": total_criteria,
        "verdicts": verdicts_out,
    }


__all__ = [
    "ALL_PASS_FIELD",
    "ALL_PASS_RATE_FIELD",
    "CRITERION_PASS_RATE_FIELD",
    "DEFAULT_JUDGE_BATCH_SIZE",
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_MAX_DELIVERABLE_CHARS",
    "BatchJudge",
    "build_rubric_judge",
    "score_rubric",
]
