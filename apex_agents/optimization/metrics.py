"""Field configs, scoring, and the MetricsCalculator for APEX-Agents.

A single optimization field, ``rubric_pass_rate``, scores the agent's
final answer against the task's rubric. Each rubric criterion is
graded binary Met / Not-met by an LLM judge (Mercor's Archipelago
default is ``gemini/gemini-2.5-flash`` — the "output_llm" verifier).
The task score is the fraction of criteria Met; the benchmark metric
is the mean task score over the evaluated tasks.

The runtime owns the judge call (it has the agent's final answer +
the task prompt + the rubric), so by the time scoring runs the per-case
``rubric_pass_rate`` float is already on the result. This module just
plumbs the precomputed value through the rilixai metrics protocol —
mirroring SWE-bench's ``_coerce_resolved`` / precomputed pattern.

:func:`build_rubric_judge` returns the callable the runtime uses; its
``llm`` is injectable so tests stub the verdicts and zero network
fires.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rilixai.prompt_optimization.protocols import ErrorOutput, FieldConfig, MetricsResult

from ..data.dataset import _APEX_AGENTS_GROUND_TRUTH_KEY


logger = logging.getLogger(__name__)


__all__ = [
    "APEX_AGENTS_FIELD_WEIGHTS",
    "DEFAULT_JUDGE_MODEL",
    "RUBRIC_FIELD",
    "ApexAgentsFieldConfig",
    "ApexAgentsMetricsCalculator",
    "ApexAgentsMetricsResult",
    "RubricJudge",
    "build_apex_agents_field_extractor",
    "build_rubric_judge",
    "score_rubric",
]


RUBRIC_FIELD = "rubric_pass_rate"

APEX_AGENTS_FIELD_WEIGHTS: dict[str, float] = {
    # Single continuous field in [0, 1]: the fraction of rubric
    # criteria the LLM judge marked Met. The optimizer's weighted
    # objective collapses to the mean rubric pass rate.
    RUBRIC_FIELD: 1.0,
}

DEFAULT_JUDGE_MODEL = "gemini/gemini-2.5-flash"
# Bound each judge call so a hung provider request fails the criterion
# (conservatively Not-met) fast instead of stalling the whole run.
DEFAULT_JUDGE_TIMEOUT_S = 120.0
DEFAULT_JUDGE_NUM_RETRIES = 2


# Callable contract: ``judge(criterion, answer, task_prompt) -> bool``.
RubricJudge = Callable[[str, str, str], bool]


@dataclass
class ApexAgentsFieldConfig:
    """Concrete FieldConfig for APEX-Agents' single field.

    Non-frozen so the dataclass attributes match the settable
    :class:`FieldConfig` protocol.
    """

    field_name: str
    result_path: str | None
    ground_truth_path: str | None


@dataclass
class ApexAgentsMetricsResult:
    """Concrete MetricsResult for APEX-Agents aggregate scoring."""

    field_accuracies: Mapping[str, float]
    field_sample_counts: Mapping[str, int]


def _resolve_path(obj: Any, path: str | None) -> Any:
    """Resolve a single-segment path on a dict / namespace / model object."""
    if obj is None or path is None:
        return None
    if isinstance(obj, ErrorOutput):
        return None
    if isinstance(obj, Mapping):
        return obj.get(path)
    return getattr(obj, path, None)


def _coerce_pass_rate(value: Any) -> float:
    """Coerce a ``rubric_pass_rate`` field value into a [0, 1] float."""
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
            return _coerce_pass_rate(float(value))
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
    if last.startswith("NOT MET") or last.startswith("NOTMET") or last in {"NO", "FALSE", "FAIL", "FAILED"}:
        return False
    if last == "MET" or last.startswith("MET") or last in {"YES", "TRUE", "PASS", "PASSED", "SATISFIED"}:
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


class ApexAgentsMetricsCalculator:
    """MetricsCalculator for the APEX-Agents ``rubric_pass_rate`` field."""

    def __init__(self) -> None:
        self.field_configs: list[FieldConfig] = [
            ApexAgentsFieldConfig(
                field_name=RUBRIC_FIELD,
                result_path=RUBRIC_FIELD,
                # The full rubric bundle stays attached to the case for
                # feedback / debugging consumers; the per-case
                # comparator only needs the precomputed pass-rate float
                # the runtime stashes on the result.
                ground_truth_path=_APEX_AGENTS_GROUND_TRUTH_KEY,
            ),
        ]

    def _has_valid_sample_for_comparison(
        self,
        predicted: Any,
        actual: Any,
        cfg: FieldConfig,
    ) -> bool:
        """Every scoreable APEX-Agents case carries a non-empty rubric."""
        if not isinstance(actual, Mapping):
            return False
        rubric = actual.get("rubric")
        return bool(rubric)

    def _get_comparison_method(self, cfg: FieldConfig) -> Callable[[Any, Any], float]:
        """Return the per-case comparator.

        The comparator reads the precomputed ``rubric_pass_rate`` float
        the runtime stashed on the result (the judge already ran).
        ``actual`` is the bundle — present for feedback consumers.
        """

        def _comparator(predicted: Any, actual: Any) -> float:
            return _coerce_pass_rate(predicted)

        return _comparator

    def calculate_metrics(
        self,
        results: Mapping[str, Any],
        ground_truth: Mapping[str, Mapping[str, Any]],
    ) -> MetricsResult:
        """Aggregate per-case rubric pass rates into the field accuracy."""
        cfg = self.field_configs[0]
        total = 0.0
        count = 0
        for case_key, expected in ground_truth.items():
            bundle = expected.get(_APEX_AGENTS_GROUND_TRUTH_KEY)
            source: Mapping[str, Any] = bundle if isinstance(bundle, Mapping) else expected
            rubric = source.get("rubric")
            if not rubric:
                continue
            result = results.get(case_key)
            predicted_value = _resolve_path(result, cfg.result_path)
            total += _coerce_pass_rate(predicted_value)
            count += 1

        accuracy = (total / count) if count > 0 else 0.0
        return ApexAgentsMetricsResult(
            field_accuracies={cfg.field_name: accuracy},
            field_sample_counts={cfg.field_name: count},
        )


def build_apex_agents_field_extractor() -> Callable[[Any, str], Any]:
    """Return the FieldExtractor used by the adapter for APEX-Agents cases."""

    def _extractor(obj: Any, path: str) -> Any:
        return _resolve_path(obj, path)

    return _extractor
