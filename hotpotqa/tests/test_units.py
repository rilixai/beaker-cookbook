"""HotpotQA unit tests: dataset normalization + metrics scoring."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from rilixai import Case, CaseResult

from hotpotqa.data.dataset import (
    HotpotQAParagraph,
    HotpotQARecord,
    cases_from_records,
    record_to_case,
)
from hotpotqa.optimization.metrics import (
    ANSWER_F1_FIELD,
    ANSWER_FIELD,
    RETRIEVED_TITLES_KEY,
    SUPPORTING_TITLES_RECALL_FIELD,
    HotpotQAScorer,
    f1_score,
    normalize_answer,
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


def test_cases_from_records_normalizes_hf_shape() -> None:
    cases = cases_from_records([_HF_RECORD])
    assert len(cases) == 1
    case = cases[0]
    assert isinstance(case, Case)
    assert case.case_id == "case-1"
    record = case.input
    assert isinstance(record, HotpotQARecord)
    assert record.question == "Which city has the Eiffel Tower?"
    assert record.answer == "Paris"
    assert record.question_type == "bridge"
    assert record.level == "easy"
    assert [p.title for p in record.paragraphs] == ["Eiffel Tower", "Paris", "Berlin"]
    assert record.paragraphs[0].sentences == ("The Eiffel Tower is in Paris.", "It is iron.")
    assert record.supporting_titles == ("Eiffel Tower", "Paris")
    assert record.supporting_sentence_ids == {"Eiffel Tower": (0,), "Paris": (2,)}


def test_record_to_case_exposes_ground_truth_fields_for_metrics() -> None:
    cases = cases_from_records([_HF_RECORD])
    case = cases[0]
    assert case.ground_truth["answer"] == "Paris"
    assert case.ground_truth["supporting_titles"] == ["Eiffel Tower", "Paris"]
    assert case.metadata["question_type"] == "bridge"
    assert case.metadata["num_paragraphs"] == 3
    # The group key should reflect the question type so the failure-focused
    # sampler can stratify if needed.
    assert case.group_key == "bridge"


def test_cases_from_records_accepts_pre_normalized_records() -> None:
    record = HotpotQARecord(
        case_id="rec-7",
        question="Q?",
        answer="A",
        question_type="comparison",
        level="medium",
        paragraphs=(HotpotQAParagraph(title="T", sentences=("S1",)),),
        supporting_titles=("T",),
    )
    cases = cases_from_records([record])
    assert cases[0].case_id == "rec-7"
    assert cases[0].input is record
    assert cases[0].group_key == "comparison"


def test_cases_from_records_rejects_unsupported_items() -> None:
    with pytest.raises(TypeError):
        cases_from_records([object()])  # type: ignore[list-item]


class _FakeHFDataset:
    """Tiny stand-in for an HF ``Dataset`` — just ``len`` + integer indexing.

    The paper-faithful loader needs ``len(dataset)`` to compute the
    partition slice boundaries and ``dataset[i]`` to materialize sampled
    rows. Both are 1-line on a list-backed wrapper.
    """

    def __init__(self, records: list[dict]) -> None:
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        return self._records[idx]


def _synthetic_record(idx: int) -> dict:
    """Minimal HF-shaped HotpotQA record so cases_from_records is happy."""
    return {
        "id": f"row-{idx:05d}",
        "question": f"Q{idx}?",
        "answer": f"A{idx}",
        "type": "bridge",
        "level": "easy",
        "supporting_facts": {"title": [f"T{idx}"], "sent_id": [0]},
        "context": {"title": [f"T{idx}"], "sentences": [[f"S{idx}"]]},
    }


def test_load_hotpotqa_paper_split_matches_artifact_slicing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bit-faithful reproduction of the artifact's data pipeline.

    The artifact takes the HotpotQA train split, slices fractionally
    (test = [0, 40%), val = [40%, 80%), train = [80%, 100%)), then
    samples each slice with ``random.Random(1).sample(slice, size)``.
    Verifies that ``load_hotpotqa_paper_split`` returns exactly the rows
    a hand-rolled emulation of that pipeline returns.
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

    # Reproduce the artifact's selection by hand for each partition.
    # Slice fractions match `_PAPER_PARTITION_BOUNDS`; seed=1 matches
    # `_PAPER_SAMPLE_SEED`. The artifact uses
    # `random.Random(1).sample(slice_items, k)`, which under fixed seed
    # picks the same *positions* as `random.Random(1).sample(range(n), k)`.
    def _expected(partition: str, size: int) -> list[str]:
        bounds = {"test": (0, 40), "validation": (40, 80), "train": (80, 100)}
        lo, hi = bounds[partition]
        slice_size = hi - lo
        local = _random.Random(1).sample(range(slice_size), size)
        return [fake_records[lo + i]["id"] for i in local]

    for partition, size in (("test", 12), ("validation", 12), ("train", 8)):
        cases = dataset_module.load_hotpotqa_paper_split(
            partition,
            max_cases=size,
            config="fullwiki",
        )
        assert [c.case_id for c in cases] == _expected(partition, size), (
            f"Partition {partition!r} disagrees with the artifact's "
            f"`Benchmark.trim_dataset` selection — split logic has drifted."
        )


def test_load_hotpotqa_paper_split_rejects_unknown_partition() -> None:
    from hotpotqa.data.dataset import load_hotpotqa_paper_split

    with pytest.raises(ValueError, match="partition must be one of"):
        load_hotpotqa_paper_split("dev", max_cases=10)


def test_record_to_case_explicit_group_key_wins() -> None:
    record = HotpotQARecord(
        case_id="rec-2",
        question="Q?",
        answer="A",
        question_type="bridge",
        level="easy",
        paragraphs=(),
        supporting_titles=(),
    )
    case = record_to_case(record, group_key="train-split")
    assert case.group_key == "train-split"


# ─── metrics ────────────────────────────────────────────────────────────


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


def _gt_case(*, answer: str, titles: list[str]) -> Case:
    """Build a Case whose ground_truth carries the gold answer + titles.

    ``record_to_case`` bundles ``answer`` + ``supporting_titles`` onto
    ``case.ground_truth`` — exactly the keys :class:`HotpotQAScorer`
    scores against.
    """
    record = HotpotQARecord(
        case_id="case",
        question="q?",
        answer=answer,
        question_type="bridge",
        level="easy",
        paragraphs=(),
        supporting_titles=tuple(titles),
    )
    return record_to_case(record)


def _score(*, output: dict, answer: str, titles: list[str]) -> Any:
    """Score one run_case ``output`` dict via :class:`HotpotQAScorer`."""
    scorer = HotpotQAScorer()
    return asyncio.run(
        scorer.score_case(case=_gt_case(answer=answer, titles=titles), result=CaseResult(output=output))
    )


def test_scorer_scores_em_f1_and_recall_per_case() -> None:
    # Case A: perfect answer + both gold titles retrieved.
    score_a = _score(
        output={ANSWER_FIELD: "Eiffel Tower", RETRIEVED_TITLES_KEY: ["Eiffel Tower", "Paris"]},
        answer="Eiffel Tower",
        titles=["Eiffel Tower", "Paris"],
    )
    assert score_a.field_scores[ANSWER_FIELD] == pytest.approx(1.0)
    assert score_a.field_scores[ANSWER_F1_FIELD] == pytest.approx(1.0)
    assert score_a.field_scores[SUPPORTING_TITLES_RECALL_FIELD] == pytest.approx(1.0)
    # Objective collapses to pure EM (default field weights).
    assert score_a.objective == pytest.approx(1.0)

    # Case B: wrong answer (no token overlap) + zero gold titles retrieved.
    score_b = _score(
        output={ANSWER_FIELD: "wrong answer", RETRIEVED_TITLES_KEY: ["Some Other Page"]},
        answer="Statue of Liberty",
        titles=["Statue of Liberty"],
    )
    assert score_b.field_scores[ANSWER_FIELD] == pytest.approx(0.0)
    assert score_b.field_scores[ANSWER_F1_FIELD] == pytest.approx(0.0)
    assert score_b.field_scores[SUPPORTING_TITLES_RECALL_FIELD] == pytest.approx(0.0)
    assert score_b.objective == pytest.approx(0.0)


def test_scorer_skips_fields_without_supervised_signal() -> None:
    score = _score(
        output={ANSWER_FIELD: "anything", RETRIEVED_TITLES_KEY: []},
        answer="",
        titles=[],
    )
    assert ANSWER_FIELD not in score.field_scores
    assert ANSWER_F1_FIELD not in score.field_scores
    assert SUPPORTING_TITLES_RECALL_FIELD not in score.field_scores


def test_supporting_titles_recall_is_case_insensitive() -> None:
    score = _score(
        output={ANSWER_FIELD: "x", RETRIEVED_TITLES_KEY: ["EIFFEL TOWER", "paris"]},
        answer="x",
        titles=["Eiffel Tower", "Paris"],
    )
    assert score.field_scores[SUPPORTING_TITLES_RECALL_FIELD] == pytest.approx(1.0)


# ─── rilixai Modal sandbox @spec wiring ─────────────────────────────────


def test_sandbox_spec_factory_is_registered() -> None:
    """Lock the @spec(...) registration contract so ``rilixai push`` discovery can't silently break.

    ``rilixai push --target hotpotqa/optimization/spec.py --name hotpotqa-agent``
    enumerates ``@spec``-decorated factories via the ``__rilixai_spec__``
    attribute the decorator stamps. If anyone renames ``build_spec`` or
    drops the decorator, this test fails loudly before a stale-image
    push reaches the build worker.
    """
    from rilixai.spec import get_registration

    from hotpotqa.optimization.spec import build_spec

    reg = get_registration(build_spec)
    assert reg is not None, "build_spec must carry @spec registration"
    assert reg.name == "hotpotqa-agent"
    assert reg.entrypoint == "hotpotqa.optimization.spec:build_spec"
    assert reg.metadata.get("benchmark") == "hotpotqa"
    assert reg.metadata.get("agent_kind") == "pydantic_ai"
    # Intentionally no version assertion: ``@spec`` doesn't pin a version.
    # ``sandbox.py --build`` supplies ``v<short_sha>`` at push time and
    # promotes to ``@production``. ``reg.version`` will be whatever
    # rilixai's ``DEFAULT_SPEC_VERSION`` constant is (currently ``"v1"``),
    # but that value never reaches the spec_versions table for normal
    # ``--build`` flows because the CLI ``--version`` flag overrides it.
