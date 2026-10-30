"""Deterministic metrics. Failed requests count as incorrect predictions."""

from __future__ import annotations

import math

from pydantic import BaseModel

from jev_banking77.agent import LABELS, Prediction


class Outcome(BaseModel):
    id: str
    expected: str
    prediction: Prediction | None = None
    error: str | None = None


def summarize(outcomes: list[Outcome]) -> dict[str, float | int | None]:
    if not outcomes:
        raise ValueError("Cannot score an empty evaluation")
    successful = [r for r in outcomes if r.prediction is not None]
    correct = sum(r.prediction is not None and r.prediction.intent == r.expected for r in outcomes)
    f1s = []
    for label in LABELS:
        tp = sum(r.expected == label and r.prediction is not None and r.prediction.intent == label for r in outcomes)
        fp = sum(r.expected != label and r.prediction is not None and r.prediction.intent == label for r in outcomes)
        fn = sum(r.expected == label and (r.prediction is None or r.prediction.intent != label) for r in outcomes)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(10)]
    brier = 0.0
    latencies = []
    input_tokens = 0
    usage_complete = bool(successful) and len(successful) == len(outcomes)
    for row in successful:
        prediction = row.prediction
        assert prediction is not None
        confidence = prediction.probabilities[prediction.intent]
        bins[min(int(confidence * 10), 9)].append((confidence, prediction.intent == row.expected))
        brier += sum((p - int(label == row.expected)) ** 2 for label, p in prediction.probabilities.items())
        latencies.append(prediction.latency_ms)
        usage_complete &= prediction.input_tokens is not None
        input_tokens += prediction.input_tokens or 0
    latencies.sort()
    count = len(successful)
    ece = sum(abs(sum(p - int(correct) for p, correct in bucket)) for bucket in bins) / count if count else None
    return {
        "cases": len(outcomes),
        "errors": len(outcomes) - count,
        "accuracy": correct / len(outcomes),
        "macro_f1_77_labels": sum(f1s) / len(LABELS),
        "calibration_cases": count,
        "brier_score": brier / count if count else None,
        "ece_10_bins": ece,
        "latency_p50_ms": latencies[math.ceil(count * 0.50) - 1] if count else None,
        "latency_p95_ms": latencies[math.ceil(count * 0.95) - 1] if count else None,
        "observed_input_tokens": input_tokens,
        "estimated_cost_usd": input_tokens * 0.042 / 1_000_000 if usage_complete else None,
    }
