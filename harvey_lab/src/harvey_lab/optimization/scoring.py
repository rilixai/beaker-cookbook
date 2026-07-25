"""Rubric scoring for Harvey LAB: per-criterion judge + all-pass aggregation.

Harvey LAB grades each task against an array of equally-weighted binary
criteria. A per-criterion LLM judge decides PASS / FAIL against the
criterion's ``match_criteria`` standard, seeing ONLY the deliverables that
criterion names (deliverable-scoped context — LAB tasks carry ~60 criteria,
so scoping keeps judging cost bounded). The headline benchmark metric is
**all-pass**: a task scores ``1.0`` iff every criterion passes, else ``0.0``.

Two fields are emitted per task:

* ``all_pass`` (0/1) — the LAB-AA headline metric, reported as-is.
* ``criterion_pass_rate`` (continuous) — the fraction of criteria that
  passed. This is the **optimizer objective**: all-pass is an extremely
  sparse signal on a ~60-criterion task (one missed criterion zeroes the
  whole task), so GEPA optimizes the dense per-criterion rate while all-pass
  is tracked alongside as the metric of record. See the README for the
  rationale and how to flip the objective back to all-pass.

The runtime owns the judge calls (it has the deliverables + criteria), so by
the time :class:`HarveyLabScorer` runs both floats are already on the
result; the scorer just plumbs them through the rilixai metrics protocol.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from rilixai import Case, CaseResult, CaseScore, objective_score


logger = logging.getLogger(__name__)


ALL_PASS_FIELD = "all_pass"
CRITERION_PASS_RATE_FIELD = "criterion_pass_rate"

# Weight the objective on the DENSE per-criterion rate (see module docstring).
# ``all_pass`` is scored + reported but carries no objective weight, so GEPA
# gets a usable gradient while the headline metric is still tracked.
HARVEY_LAB_FIELD_WEIGHTS: dict[str, float] = {CRITERION_PASS_RATE_FIELD: 1.0}

DEFAULT_JUDGE_MODEL = "gemini/gemini-3.5-flash"
DEFAULT_JUDGE_TIMEOUT_S = 120.0
DEFAULT_JUDGE_NUM_RETRIES = 2
DEFAULT_MAX_DELIVERABLE_CHARS = 40_000


# ``judge(task_description, criterion_title, match_criteria, agent_output) -> bool``
CriterionJudge = Callable[[str, str, str, str], bool]


# Harvey's reference judge prompt (evaluation/prompts/rubric_criterion.txt).
_JUDGE_PROMPT_TEMPLATE = """You are evaluating a legal AI agent's work product against a specific quality criterion.

## Task
{task_description}

## Agent's Output
{agent_output}

## Criterion
**{criterion_title}**

{match_criteria}

## Instructions
Evaluate the agent's output against the criterion above.
- **PASS**: The agent's output satisfies the criterion as described
- **FAIL**: The agent's output does not satisfy the criterion as described

Respond with JSON only:

```json
{{
  "verdict": "pass" | "fail",
  "reasoning": "Brief explanation"
}}
```"""


_VERDICT_TAG_RE = re.compile(r"VERDICT\s*[:\-]?\s*(PASS|FAIL|MET|NOT[\s_-]?MET)\b", re.IGNORECASE)


def _parse_verdict(text: str) -> bool:
    """Parse a judge reply into PASS (True) / FAIL (False).

    Prefers the reference JSON ``{"verdict": "pass"|"fail"}`` shape, then a
    ``VERDICT: PASS/FAIL`` tag, then the last non-empty line. Anything
    ambiguous → conservative FAIL (logged) so an unparseable judge is visible.
    """
    blob = text or ""
    for match in re.finditer(r"\{[^{}]*\"verdict\"[^{}]*\}", blob, re.DOTALL):
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            continue
        verdict = str(parsed.get("verdict", "")).strip().lower()
        if verdict in ("pass", "fail"):
            return verdict == "pass"

    tags = _VERDICT_TAG_RE.findall(blob)
    if tags:
        last = tags[-1].upper()
        return "FAIL" not in last and "NOT" not in last

    for line in reversed(blob.strip().splitlines()):
        token = line.strip().strip("*_`#.\"' \t").upper()
        if not token:
            continue
        if re.match(r"(FAIL|NOT[\s_-]?MET)\b", token) or token in {"NO", "FALSE"}:
            return False
        if re.match(r"(PASS|MET)\b", token) or token in {"YES", "TRUE", "SATISFIED"}:
            return True
        break

    logger.warning("Rubric judge reply had no parseable verdict (scoring FAIL): %r", blob[:200])
    return False


def build_criterion_judge(
    model: str = DEFAULT_JUDGE_MODEL,
    llm: Callable[..., Any] | None = None,
    *,
    timeout: float = DEFAULT_JUDGE_TIMEOUT_S,
    num_retries: int = DEFAULT_JUDGE_NUM_RETRIES,
) -> CriterionJudge:
    """Return a per-criterion ``judge(...) -> bool`` callable.

    ``llm`` is injectable for tests: pass ``llm(model=..., messages=...) ->
    str | response`` returning a scripted verdict so zero network fires. In
    production ``llm`` is ``None`` and litellm is imported lazily.
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

    def _judge(task_description: str, criterion_title: str, match_criteria: str, agent_output: str) -> bool:
        prompt = _JUDGE_PROMPT_TEMPLATE.format(
            task_description=task_description,
            agent_output=agent_output or "(the agent produced no deliverables)",
            criterion_title=criterion_title or "(untitled criterion)",
            match_criteria=match_criteria,
        )
        try:
            verdict = _call_llm([{"role": "user", "content": prompt}])
        except Exception as exc:  # pragma: no cover - judge outage is a FAIL
            logger.warning("Rubric judge call failed (scoring FAIL): %s", exc)
            return False
        return _parse_verdict(verdict)

    return _judge


def _scope_deliverables(
    criterion_deliverables: Sequence[str],
    deliverables: Mapping[str, str],
    *,
    max_chars: int,
) -> str:
    """Concatenate only the deliverables a criterion names (deliverable-scoping).

    Matches by exact filename, then by basename. A criterion that names no
    deliverable (or names ones the agent never produced) falls back to every
    produced deliverable so the judge still has context.
    """
    wanted = [str(d) for d in criterion_deliverables if d]
    selected: dict[str, str] = {}
    for name in wanted:
        if name in deliverables:
            selected[name] = deliverables[name]
            continue
        base = name.rsplit("/", 1)[-1]
        for produced, text in deliverables.items():
            if produced.rsplit("/", 1)[-1] == base:
                selected[produced] = text
    if not selected:
        selected = dict(deliverables)
    parts: list[str] = []
    for name, text in selected.items():
        parts.append(f"### {name}\n{text[:max_chars]}")
    return "\n\n".join(parts)


def score_all_pass(
    *,
    criteria: Sequence[Mapping[str, Any]],
    deliverables: Mapping[str, str],
    task_description: str,
    judge: CriterionJudge,
    max_deliverable_chars: int = DEFAULT_MAX_DELIVERABLE_CHARS,
) -> dict[str, Any]:
    """Grade every criterion and aggregate into the LAB all-pass result.

    Returns ``{"all_pass", "criterion_pass_rate", "n_total", "n_passed",
    "verdicts"}``. An empty rubric yields ``n_total == 0`` (an unscoreable
    task) — the caller signals that upstream so it is excluded from the mean.
    """
    scoreable = [c for c in criteria if str(c.get("match_criteria") or "").strip()]
    n_total = len(scoreable)
    if n_total == 0:
        return {"all_pass": 0.0, "criterion_pass_rate": 0.0, "n_total": 0, "n_passed": 0, "verdicts": []}
    n_passed = 0
    verdicts: list[dict[str, Any]] = []
    for criterion in scoreable:
        scoped = _scope_deliverables(
            criterion.get("deliverables") or (),
            deliverables,
            max_chars=max_deliverable_chars,
        )
        try:
            passed = judge(
                task_description,
                str(criterion.get("title") or ""),
                str(criterion.get("match_criteria") or ""),
                scoped,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Rubric judge raised on a criterion: %s", exc)
            passed = False
        n_passed += int(bool(passed))
        verdicts.append({"id": criterion.get("id"), "passed": bool(passed)})
    return {
        "all_pass": 1.0 if n_passed == n_total else 0.0,
        "criterion_pass_rate": n_passed / n_total,
        "n_total": n_total,
        "n_passed": n_passed,
        "verdicts": verdicts,
    }


def _bounded(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


class HarveyLabScorer:
    """rilixai :class:`CaseScorer` for the ``all_pass`` + ``criterion_pass_rate`` fields.

    The runtime precomputes both floats (it ran the per-criterion judge), so
    this scorer reads them back, clamps to ``[0, 1]``, and collapses the
    objective onto the dense ``criterion_pass_rate`` field. A ``None`` marks an
    unscoreable case (no criteria): the field map is emptied so the case is
    excluded from the aggregates instead of counting as a real ``0.0``.
    """

    def __init__(self, field_weights: Mapping[str, float] | None = None) -> None:
        self.field_weights: dict[str, float] = dict(field_weights or HARVEY_LAB_FIELD_WEIGHTS)

    async def score_case(self, *, case: Case, result: CaseResult) -> CaseScore:
        del case
        output = result.output if isinstance(result.output, Mapping) else {}
        if output.get(CRITERION_PASS_RATE_FIELD, False) is None:
            return CaseScore(field_scores={}, objective=0.0, key=ALL_PASS_FIELD)
        field_scores = {
            ALL_PASS_FIELD: _bounded(output.get(ALL_PASS_FIELD)),
            CRITERION_PASS_RATE_FIELD: _bounded(output.get(CRITERION_PASS_RATE_FIELD)),
        }
        return CaseScore(
            field_scores=field_scores,
            objective=objective_score(field_scores, field_weights=self.field_weights),
            key=ALL_PASS_FIELD,
        )


__all__ = [
    "ALL_PASS_FIELD",
    "CRITERION_PASS_RATE_FIELD",
    "DEFAULT_JUDGE_MODEL",
    "HARVEY_LAB_FIELD_WEIGHTS",
    "CriterionJudge",
    "HarveyLabScorer",
    "build_criterion_judge",
    "score_all_pass",
]
