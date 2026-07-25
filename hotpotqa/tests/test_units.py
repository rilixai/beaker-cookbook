"""HotpotQA unit tests: dataset normalization, splits, and scoring."""

from __future__ import annotations

import pytest

from hotpotqa.data.dataset import (
    HotpotQAParagraph,
    HotpotQARecord,
    records_from_raw,
)
from hotpotqa.evaluation.scoring import (
    ANSWER_F1_FIELD,
    ANSWER_FIELD,
    HOTPOTQA_FIELD_WEIGHTS,
    SUPPORTING_TITLES_RECALL_FIELD,
    f1_score,
    normalize_answer,
    objective_score,
    score_prediction,
)


# ─── dataset ────────────────────────────────────────────────────────────


_HF_RECORD = {
    "id": "case-1",
    "question": "Which city has the Eiffel Tower?",
    "answer": "Paris",
    "type": "bridge",
    "level": "easy",
    "supporting_facts": {"title": ["Eiffel Tower", "Paris"], "sent_id": [0, 2]},
    "context": {
        "title": ["Eiffel Tower", "Paris", "Berlin"],
        "sentences": [
            ["The Eiffel Tower is in Paris.", "It is iron."],
            ["Paris is a city in France.", "Paris has many monuments.", "It also has the Eiffel Tower."],
            ["Berlin is a city in Germany."],
        ],
    },
}


def test_records_from_raw_normalizes_hf_shape() -> None:
    records = records_from_raw([_HF_RECORD])
    assert len(records) == 1
    record = records[0]
    assert isinstance(record, HotpotQARecord)
    assert record.case_id == "case-1"
    assert record.question == "Which city has the Eiffel Tower?"
    assert record.answer == "Paris"
    assert record.question_type == "bridge"
    assert record.level == "easy"
    assert [p.title for p in record.paragraphs] == ["Eiffel Tower", "Paris", "Berlin"]
    assert record.paragraphs[0].sentences == ("The Eiffel Tower is in Paris.", "It is iron.")
    assert record.supporting_titles == ("Eiffel Tower", "Paris")
    assert record.supporting_sentence_ids == {"Eiffel Tower": (0,), "Paris": (2,)}


def test_records_from_raw_accepts_pre_normalized_records() -> None:
    record = HotpotQARecord(
        case_id="rec-7",
        question="Q?",
        answer="A",
        question_type="comparison",
        level="medium",
        paragraphs=(HotpotQAParagraph(title="T", sentences=("S1",)),),
        supporting_titles=("T",),
    )
    assert records_from_raw([record])[0] is record


def test_records_from_raw_rejects_unsupported_items() -> None:
    with pytest.raises(TypeError):
        records_from_raw([object()])  # type: ignore[list-item]


class _FakeHFDataset:
    """Tiny stand-in for an HF ``Dataset`` — just ``len`` + integer indexing.

    The split loader needs ``len(dataset)`` to compute the partition slice
    boundaries and ``dataset[i]`` to materialize sampled rows. Both are
    1-line on a list-backed wrapper.
    """

    def __init__(self, records: list[dict]) -> None:
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        return self._records[idx]


def _synthetic_record(idx: int) -> dict:
    """Minimal HF-shaped HotpotQA record so records_from_raw is happy."""
    return {
        "id": f"row-{idx:05d}",
        "question": f"Q{idx}?",
        "answer": f"A{idx}",
        "type": "bridge",
        "level": "easy",
        "supporting_facts": {"title": [f"T{idx}"], "sent_id": [0]},
        "context": {"title": [f"T{idx}"], "sentences": [[f"S{idx}"]]},
    }


def test_load_hotpotqa_paper_split_matches_reference_slicing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bit-faithful reproduction of the reference data pipeline.

    It takes the HotpotQA train split, slices fractionally (test = [0, 40%),
    val = [40%, 80%), train = [80%, 100%)), then samples each slice with
    ``random.Random(1).sample(slice, size)``. Verifies that
    ``load_hotpotqa_paper_split`` returns exactly the rows a hand-rolled
    emulation of that pipeline returns.
    """
    import random as _random

    from hotpotqa.data import dataset as dataset_module

    fake_records = [_synthetic_record(i) for i in range(100)]
    fake_dataset = _FakeHFDataset(fake_records)

    def _fake_load_dataset(name: str, config: str, **kwargs: object) -> _FakeHFDataset:
        assert name == "hotpotqa/hotpot_qa"
        assert config == "fullwiki"
        assert kwargs.get("split") == "train"
        return fake_dataset

    import sys
    import types as _types

    fake_module = _types.ModuleType("datasets")
    fake_module.load_dataset = _fake_load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    # Reproduce the selection by hand for each partition. Slice fractions
    # match `_PAPER_PARTITION_BOUNDS`; seed=1 matches `_PAPER_SAMPLE_SEED`.
    # ``random.Random(1).sample(slice_items, k)`` under a fixed seed picks
    # the same *positions* as ``random.Random(1).sample(range(n), k)``.
    def _expected(partition: str, size: int) -> list[str]:
        bounds = {"test": (0, 40), "validation": (40, 80), "train": (80, 100)}
        lo, hi = bounds[partition]
        slice_size = hi - lo
        local = _random.Random(1).sample(range(slice_size), size)
        return [fake_records[lo + i]["id"] for i in local]

    for partition, size in (("test", 12), ("validation", 12), ("train", 8)):
        records = dataset_module.load_hotpotqa_paper_split(
            partition,
            max_cases=size,
            config="fullwiki",
        )
        assert [r.case_id for r in records] == _expected(partition, size), (
            f"Partition {partition!r} disagrees with the reference selection — split logic has drifted."
        )


def test_load_hotpotqa_paper_split_rejects_unknown_partition() -> None:
    from hotpotqa.data.dataset import load_hotpotqa_paper_split

    with pytest.raises(ValueError, match="partition must be one of"):
        load_hotpotqa_paper_split("dev", max_cases=10)


# ─── scoring ────────────────────────────────────────────────────────────


def test_normalize_answer_strips_articles_punctuation_case() -> None:
    assert normalize_answer("The Eiffel Tower!") == "eiffel tower"
    assert normalize_answer("  a   Quick   brown  fox.") == "quick brown fox"
    assert normalize_answer(None) == ""


def test_f1_score_token_overlap() -> None:
    assert f1_score("the eiffel tower", "Eiffel Tower") == pytest.approx(1.0)
    assert 0.4 < f1_score("Paris France", "France") < 0.8
    assert f1_score("yes", "no") == 0.0
    assert f1_score("yes", "yes") == 1.0
    # Official HotpotQA scorer: empty answers get zero F1 (no token overlap,
    # not a yes/no/noanswer special case) — matches hotpot_evaluate_v1.py.
    assert f1_score("", "") == 0.0
    assert f1_score("anything", "") == 0.0


def test_f1_score_yes_no_short_circuits_to_exact_match() -> None:
    # Token F1 of two single-token answers can still hit 0.0 by coincidence,
    # but the explicit short-circuit guarantees yes/no questions are scored
    # by exact match — never by partial token overlap with a longer answer.
    assert f1_score("yes", "no") == 0.0
    assert f1_score("yes definitely", "yes") == 0.0
    assert f1_score("yes", "yes") == 1.0
    assert f1_score("noanswer", "noanswer") == 1.0


def _record(*, answer: str, titles: list[str]) -> HotpotQARecord:
    return HotpotQARecord(
        case_id="case",
        question="q?",
        answer=answer,
        question_type="bridge",
        level="easy",
        paragraphs=(),
        supporting_titles=tuple(titles),
    )


def test_score_prediction_scores_em_f1_and_recall_per_case() -> None:
    # Case A: perfect answer + both gold titles retrieved.
    scores_a = score_prediction(
        record=_record(answer="Eiffel Tower", titles=["Eiffel Tower", "Paris"]),
        answer="Eiffel Tower",
        retrieved_titles=["Eiffel Tower", "Paris"],
    )
    assert scores_a[ANSWER_FIELD] == pytest.approx(1.0)
    assert scores_a[ANSWER_F1_FIELD] == pytest.approx(1.0)
    assert scores_a[SUPPORTING_TITLES_RECALL_FIELD] == pytest.approx(1.0)
    # The objective collapses to pure EM under the default field weights.
    assert objective_score(scores_a, HOTPOTQA_FIELD_WEIGHTS) == pytest.approx(1.0)

    # Case B: wrong answer (no token overlap) + zero gold titles retrieved.
    scores_b = score_prediction(
        record=_record(answer="Statue of Liberty", titles=["Statue of Liberty"]),
        answer="wrong answer",
        retrieved_titles=["Some Other Page"],
    )
    assert scores_b[ANSWER_FIELD] == pytest.approx(0.0)
    assert scores_b[ANSWER_F1_FIELD] == pytest.approx(0.0)
    assert scores_b[SUPPORTING_TITLES_RECALL_FIELD] == pytest.approx(0.0)
    assert objective_score(scores_b, HOTPOTQA_FIELD_WEIGHTS) == pytest.approx(0.0)


def test_score_prediction_skips_fields_without_supervised_signal() -> None:
    scores = score_prediction(
        record=_record(answer="", titles=[]),
        answer="anything",
        retrieved_titles=[],
    )
    assert scores == {}


def test_supporting_titles_recall_is_case_insensitive() -> None:
    scores = score_prediction(
        record=_record(answer="x", titles=["Eiffel Tower", "Paris"]),
        answer="x",
        retrieved_titles=["EIFFEL TOWER", "paris"],
    )
    assert scores[SUPPORTING_TITLES_RECALL_FIELD] == pytest.approx(1.0)


def test_objective_score_weighting() -> None:
    scores = {ANSWER_FIELD: 0.0, ANSWER_F1_FIELD: 1.0}
    # Unweighted: every scored field counts equally.
    assert objective_score(scores) == pytest.approx(0.5)
    # A weighted field the case did not score counts as zero.
    assert objective_score({ANSWER_F1_FIELD: 1.0}, HOTPOTQA_FIELD_WEIGHTS) == pytest.approx(0.0)
    # Weights are normalized, so they need not sum to 1.
    assert objective_score(scores, {ANSWER_FIELD: 2.0, ANSWER_F1_FIELD: 2.0}) == pytest.approx(0.5)
    # All-zero weights leave nothing measurable.
    assert objective_score(scores, {ANSWER_FIELD: 0.0}) == pytest.approx(0.0)
