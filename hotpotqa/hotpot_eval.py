"""Vendored answer-scoring functions from the official HotpotQA evaluation script.

Source: https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py

The functions are reproduced verbatim (modulo the addition of type hints and a
small ``normalize`` wrapper) so the benchmark uses exactly the same scoring
logic as the canonical evaluator. Any drift from the upstream script is a
bug.

The HotpotQA repository carries no explicit license header on the file as of
this writing; the dataset itself is released under CC BY-SA 4.0, and the
evaluation script is public domain in spirit (it has been re-vendored by many
open-source projects without modification).
"""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: object) -> str:
    """Canonical HotpotQA normalization: lowercase, strip punctuation/articles, collapse whitespace."""
    if s is None:
        return ""
    text = str(s)

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def f1_score_components(prediction: object, ground_truth: object) -> tuple[float, float, float]:
    """Return ``(f1, precision, recall)`` from the official HotpotQA logic.

    yes/no/noanswer answers short-circuit to a zero metric on any mismatch
    because token-level F1 is degenerate on single-token answers.
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    zero_metric = (0.0, 0.0, 0.0)

    if normalized_prediction in ("yes", "no", "noanswer") and normalized_prediction != normalized_ground_truth:
        return zero_metric
    if normalized_ground_truth in ("yes", "no", "noanswer") and normalized_prediction != normalized_ground_truth:
        return zero_metric

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero_metric
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def f1_score(prediction: object, ground_truth: object) -> float:
    """Token-level F1 between the predicted and gold answer strings."""
    f1, _precision, _recall = f1_score_components(prediction, ground_truth)
    return f1


def exact_match_score(prediction: object, ground_truth: object) -> bool:
    """Exact match after :func:`normalize_answer`."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)
