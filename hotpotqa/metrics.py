"""Field configs, scoring, and the MetricsCalculator for HotpotQA.

Three optimization fields are exposed:

* ``answer`` — exact-match (0/1) after HotpotQA-standard normalization.
* ``answer_f1`` — token-level F1 (the headline HotpotQA answer metric).
* ``supporting_titles_recall`` — fraction of gold supporting paragraph titles
  the pipeline retrieved at any hop. This is the multi-hop signal the
  reflection LM uses to learn when retrieval is missing hops.

The HotpotQA paper also reports supporting-fact F1 at the sentence level. We
intentionally score at title granularity here because the pipeline retrieves
paragraphs, not sentences; sentence-level supporting-fact extraction can be
layered on later without changing the optimization objective.

The answer-string scoring functions (``normalize_answer`` and ``f1_score``)
are re-exported from :mod:`.hotpot_eval`, which vendors the canonical
HotpotQA evaluator. Keep them as the single source of truth.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rilixai.prompt_optimization.protocols import ErrorOutput, FieldConfig, MetricsResult

from .hotpot_eval import exact_match_score, f1_score, normalize_answer


__all__ = [
    "ANSWER_F1_FIELD",
    "ANSWER_FIELD",
    "HOTPOTQA_FIELD_WEIGHTS",
    "SUPPORTING_TITLES_RECALL_FIELD",
    "HotpotQAFieldConfig",
    "HotpotQAMetricsCalculator",
    "HotpotQAMetricsResult",
    "build_hotpotqa_field_extractor",
    "f1_score",
    "normalize_answer",
    "supporting_title_set",
]


ANSWER_FIELD = "answer"
ANSWER_F1_FIELD = "answer_f1"
SUPPORTING_TITLES_RECALL_FIELD = "supporting_titles_recall"


HOTPOTQA_FIELD_WEIGHTS: dict[str, float] = {
    # Paper-faithful pure exact-match. The GEPA artifact registers
    # HotpotQA with ``dspy.evaluate.answer_exact_match`` at ``frac=1.0``
    # — every per-case reward GEPA sees is the EM 0/1. We match that here
    # so optimize-time selection pressure is identical to the paper's.
    #
    # ``answer_f1`` and ``supporting_titles_recall`` are still *computed*
    # and surfaced in the eval summary as diagnostics — they just don't
    # influence candidate selection. To run the diagnostics-blended
    # variant for ablations, override ``field_weights`` when calling
    # :func:`build_hotpotqa_spec`.
    ANSWER_FIELD: 1.0,
    ANSWER_F1_FIELD: 0.0,
    SUPPORTING_TITLES_RECALL_FIELD: 0.0,
}


@dataclass
class HotpotQAFieldConfig:
    """Concrete FieldConfig implementation for HotpotQA fields.

    Intentionally non-frozen so the dataclass's attributes are settable —
    the :class:`FieldConfig` protocol declares them as plain (settable)
    class attributes, and a frozen dataclass would fail structural-typing
    checks on the protocol.
    """

    field_name: str
    result_path: str | None
    ground_truth_path: str | None


@dataclass
class HotpotQAMetricsResult:
    """Concrete MetricsResult for HotpotQA aggregate scoring."""

    field_accuracies: Mapping[str, float]
    field_sample_counts: Mapping[str, int]


def _exact_match(predicted: Any, actual: Any) -> float:
    return 1.0 if exact_match_score(predicted, actual) else 0.0


def _supporting_titles_recall(predicted: Any, actual: Any) -> float:
    gold_titles = _coerce_title_sequence(actual)
    if not gold_titles:
        return 1.0
    retrieved_titles = {t.strip().lower() for t in _coerce_title_sequence(predicted) if t and t.strip()}
    if not retrieved_titles:
        return 0.0
    hits = sum(1 for gold in gold_titles if gold.strip().lower() in retrieved_titles)
    return hits / len(gold_titles)


def _coerce_title_sequence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        # Permit the HF ``supporting_facts`` shape ``{"title": [...], "sent_id": [...]}``.
        titles = value.get("title")
        if isinstance(titles, Sequence) and not isinstance(titles, (str, bytes, bytearray)):
            return [str(t) for t in titles]
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(t) for t in value]
    return [str(value)]


class HotpotQAMetricsCalculator:
    """MetricsCalculator implementation for HotpotQA fields."""

    def __init__(self) -> None:
        # Annotate as `list[FieldConfig]` so the calculator structurally
        # matches the optimizer's MetricsCalculator protocol (which is
        # invariant in the field-config type).
        self.field_configs: list[FieldConfig] = [
            HotpotQAFieldConfig(
                field_name=ANSWER_FIELD,
                result_path=ANSWER_FIELD,
                ground_truth_path=ANSWER_FIELD,
            ),
            HotpotQAFieldConfig(
                field_name=ANSWER_F1_FIELD,
                # The pipeline produces a single answer string; both EM and F1
                # read from the same predicted/expected slot.
                result_path=ANSWER_FIELD,
                ground_truth_path=ANSWER_FIELD,
            ),
            HotpotQAFieldConfig(
                field_name=SUPPORTING_TITLES_RECALL_FIELD,
                result_path="retrieved_titles",
                ground_truth_path="supporting_titles",
            ),
        ]

    def calculate_metrics(
        self,
        results: Mapping[str, Any],
        ground_truth: Mapping[str, Mapping[str, Any]],
    ) -> MetricsResult:
        """Aggregate per-case scores into field accuracies and sample counts.

        Errored cases (sentinel :class:`ErrorOutput`) contribute zero to every
        field but still increment the sample count, so dataset-level accuracy
        cannot be inflated by silently dropping failures.
        """
        totals = {cfg.field_name: 0.0 for cfg in self.field_configs}
        counts = {cfg.field_name: 0 for cfg in self.field_configs}
        for case_key, expected in ground_truth.items():
            result = results.get(case_key)
            for cfg in self.field_configs:
                comparator = self._get_comparison_method(cfg)
                predicted_value = _resolve_path(result, cfg.result_path)
                expected_value = _resolve_path(expected, cfg.ground_truth_path)
                if not self._has_valid_sample_for_comparison(predicted_value, expected_value, cfg):
                    continue
                try:
                    score = float(comparator(predicted_value, expected_value))
                except Exception:
                    score = 0.0
                totals[cfg.field_name] += max(0.0, min(1.0, score))
                counts[cfg.field_name] += 1

        accuracies = {
            cfg.field_name: (totals[cfg.field_name] / counts[cfg.field_name]) if counts[cfg.field_name] > 0 else 0.0
            for cfg in self.field_configs
        }
        return HotpotQAMetricsResult(field_accuracies=accuracies, field_sample_counts=counts)

    def _has_valid_sample_for_comparison(
        self,
        predicted: Any,
        actual: Any,
        cfg: FieldConfig,
    ) -> bool:
        """Every supervised HotpotQA case has an answer and ≥1 supporting title."""
        if cfg.field_name == SUPPORTING_TITLES_RECALL_FIELD:
            return bool(_coerce_title_sequence(actual))
        return actual is not None and str(actual).strip() != ""

    def _get_comparison_method(
        self,
        cfg: FieldConfig,
    ) -> Callable[[Any, Any], float]:
        if cfg.field_name == ANSWER_FIELD:
            return _exact_match
        if cfg.field_name == ANSWER_F1_FIELD:
            return f1_score
        if cfg.field_name == SUPPORTING_TITLES_RECALL_FIELD:
            return _supporting_titles_recall
        raise KeyError(f"Unknown HotpotQA field: {cfg.field_name!r}")


def _resolve_path(obj: Any, path: str | None) -> Any:
    """Resolve a single-segment dotted path on a dict / namespace / model object.

    HotpotQA fields use flat paths only — implementing dotted nesting here
    would be dead code.
    """
    if obj is None or path is None:
        return None
    if isinstance(obj, ErrorOutput):
        return None
    if isinstance(obj, Mapping):
        return obj.get(path)
    return getattr(obj, path, None)


def build_hotpotqa_field_extractor() -> Callable[[Any, str], Any]:
    """Return the FieldExtractor used by the adapter for HotpotQA cases."""

    def _extractor(obj: Any, path: str) -> Any:
        return _resolve_path(obj, path)

    return _extractor


def supporting_title_set(records: Iterable[Mapping[str, Any]]) -> set[str]:
    """Helper for tests / debugging: collect lowercased supporting titles."""
    titles: set[str] = set()
    for record in records:
        for title in _coerce_title_sequence(record.get("supporting_facts")):
            titles.add(title.strip().lower())
    return titles
