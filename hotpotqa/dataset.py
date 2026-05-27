"""HotpotQA dataset loading and conversion to optimizer cases.

The HuggingFace ``hotpot_qa`` dataset (config ``distractor``) is the source of
truth. Each raw record has these fields:

* ``id`` — case id
* ``question`` — natural-language question
* ``answer`` — gold answer string
* ``type`` — ``"bridge"`` or ``"comparison"``
* ``level`` — ``"easy"``, ``"medium"``, or ``"hard"``
* ``supporting_facts`` — ``{"title": list[str], "sent_id": list[int]}``
* ``context`` — ``{"title": list[str], "sentences": list[list[str]]}`` (10 paragraphs)

The HF ``datasets`` library is imported lazily so this module remains testable
without it; pre-loaded dict records can be passed directly to
:func:`cases_from_records` for fixtures.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rilixai.prompt_optimization.models import Case


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HotpotQAParagraph:
    """A single distractor paragraph: title + ordered sentences."""

    title: str
    sentences: tuple[str, ...]

    @property
    def text(self) -> str:
        return " ".join(self.sentences).strip()


@dataclass(frozen=True)
class HotpotQARecord:
    """Normalized HotpotQA record used by the runtime and metrics."""

    case_id: str
    question: str
    answer: str
    question_type: str
    level: str
    paragraphs: tuple[HotpotQAParagraph, ...]
    supporting_titles: tuple[str, ...]
    supporting_sentence_ids: Mapping[str, tuple[int, ...]] = field(default_factory=dict)

    def paragraphs_by_title(self) -> dict[str, HotpotQAParagraph]:
        return {p.title: p for p in self.paragraphs}


def _normalize_supporting_facts(raw: Any) -> tuple[tuple[str, ...], dict[str, tuple[int, ...]]]:
    """Return ``(titles_in_order, {title: sentence_ids})`` from the HF shape."""
    if not isinstance(raw, Mapping):
        return (), {}
    titles_raw = raw.get("title", [])
    sent_ids_raw = raw.get("sent_id", [])
    if not isinstance(titles_raw, Sequence) or isinstance(titles_raw, (str, bytes, bytearray)):
        return (), {}
    if not isinstance(sent_ids_raw, Sequence) or isinstance(sent_ids_raw, (str, bytes, bytearray)):
        sent_ids_raw = [None] * len(titles_raw)

    ordered: list[str] = []
    seen: set[str] = set()
    by_title: dict[str, list[int]] = {}
    for title, sent_id in zip(titles_raw, sent_ids_raw):
        title_str = str(title)
        if title_str not in seen:
            seen.add(title_str)
            ordered.append(title_str)
        try:
            sid = int(sent_id) if sent_id is not None else None
        except (TypeError, ValueError):
            sid = None
        if sid is not None:
            by_title.setdefault(title_str, []).append(sid)
    return tuple(ordered), {title: tuple(ids) for title, ids in by_title.items()}


def _normalize_context(raw: Any) -> tuple[HotpotQAParagraph, ...]:
    """Convert the HF ``context`` shape into ordered paragraphs."""
    if not isinstance(raw, Mapping):
        return ()
    titles = raw.get("title", [])
    sentences_per_paragraph = raw.get("sentences", [])
    if not isinstance(titles, Sequence) or isinstance(titles, (str, bytes, bytearray)):
        return ()
    if not isinstance(sentences_per_paragraph, Sequence) or isinstance(
        sentences_per_paragraph, (str, bytes, bytearray)
    ):
        return ()

    paragraphs: list[HotpotQAParagraph] = []
    for title, sentences in zip(titles, sentences_per_paragraph):
        if not isinstance(sentences, Sequence) or isinstance(sentences, (str, bytes, bytearray)):
            sentences = []
        paragraphs.append(
            HotpotQAParagraph(
                title=str(title),
                sentences=tuple(str(s) for s in sentences),
            )
        )
    return tuple(paragraphs)


def _normalize_record(raw: Mapping[str, Any], *, index_hint: int) -> HotpotQARecord:
    case_id = str(raw.get("id") or raw.get("_id") or f"hotpot-{index_hint}")
    question = str(raw.get("question") or "").strip()
    answer = str(raw.get("answer") or "").strip()
    question_type = str(raw.get("type") or "").strip()
    level = str(raw.get("level") or "").strip()
    paragraphs = _normalize_context(raw.get("context"))
    supporting_titles, supporting_sentence_ids = _normalize_supporting_facts(raw.get("supporting_facts"))
    return HotpotQARecord(
        case_id=case_id,
        question=question,
        answer=answer,
        question_type=question_type,
        level=level,
        paragraphs=paragraphs,
        supporting_titles=supporting_titles,
        supporting_sentence_ids=supporting_sentence_ids,
    )


def record_to_case(record: HotpotQARecord, *, group_key: str | None = None) -> Case:
    """Convert a normalized HotpotQA record into a rilixai :class:`Case`.

    The ``input`` payload is the entire record (the pipeline reads question
    plus distractor paragraphs from it). ``ground_truth`` stores the keys the
    metrics calculator scores against.
    """
    resolved_group = group_key or record.question_type or "hotpotqa"
    return Case(
        input=record,
        case_id=record.case_id,
        ground_truth={
            "answer": record.answer,
            "supporting_titles": list(record.supporting_titles),
        },
        group_key=resolved_group,
        metadata={
            "level": record.level,
            "question_type": record.question_type,
            "num_paragraphs": len(record.paragraphs),
        },
    )


def cases_from_records(
    records: Iterable[Mapping[str, Any] | HotpotQARecord],
    *,
    group_key: str | None = None,
) -> list[Case]:
    """Build a list of optimizer cases from raw or normalized records."""
    cases: list[Case] = []
    for idx, raw in enumerate(records):
        if isinstance(raw, HotpotQARecord):
            record = raw
        elif isinstance(raw, Mapping):
            record = _normalize_record(raw, index_hint=idx)
        else:
            raise TypeError(
                f"cases_from_records expected Mapping or HotpotQARecord at position {idx}, got {type(raw).__name__}."
            )
        cases.append(record_to_case(record, group_key=group_key))
    return cases


def load_hotpotqa_split(
    split: str = "validation",
    *,
    max_cases: int | None = None,
    config: str = "distractor",
    cache_dir: str | None = None,
    shuffle_seed: int | None = None,
    skip: int = 0,
) -> list[Case]:
    """Load a HotpotQA split from HuggingFace and convert it to optimizer cases.

    ``shuffle_seed`` controls the partitioning. When set, all examples in
    the HF split are shuffled with a fixed-seed RNG *before* ``skip`` /
    ``max_cases`` are applied — this makes the val vs. test partition
    reproducible across runs and independent of HF's storage order
    (which could theoretically change with dataset versions). When
    ``shuffle_seed`` is ``None``, the function falls back to a sequential
    scan that preserves HF's order (the original behavior).

    ``skip`` is applied after shuffling so two callers wanting disjoint
    slices (e.g. val = ``skip=0, max_cases=300`` and test = ``skip=300,
    max_cases=300``) get reproducible partitions out of the same shuffle
    when they share ``shuffle_seed``.

    The ``datasets`` library is imported lazily so the rest of the
    benchmark can be exercised in unit tests without network access.
    Pass pre-built records to :func:`cases_from_records` for offline
    fixtures.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Loading HotpotQA from HuggingFace requires the optional `datasets` dependency. "
            "Install it with `pip install datasets`."
        ) from exc

    dataset = load_dataset("hotpot_qa", config, split=split, cache_dir=cache_dir, trust_remote_code=True)

    if shuffle_seed is not None:
        import random as _random

        indices = list(range(len(dataset)))
        _random.Random(shuffle_seed).shuffle(indices)
        if skip:
            indices = indices[skip:]
        if max_cases is not None:
            indices = indices[:max_cases]
        records: list[Mapping[str, Any]] = [dataset[i] for i in indices]
        return cases_from_records(records)

    records = []
    for idx, raw in enumerate(dataset):
        if idx < skip:
            continue
        records.append(raw)
        if max_cases is not None and len(records) >= max_cases:
            break
    return cases_from_records(records)


# Partition boundaries the GEPA artifact's ``Benchmark`` base class uses
# when it slices the HotpotQA train split into test/val/train portions
# before downsampling. See
# https://github.com/gepa-ai/gepa-artifact/blob/main/gepa_artifact/benchmarks/benchmark.py
_PAPER_PARTITION_BOUNDS: Mapping[str, tuple[float, float]] = {
    "test": (0.0, 0.4),
    "validation": (0.4, 0.8),
    "train": (0.8, 1.0),
}

# Fixed sampling seed the artifact uses in ``trim_dataset``:
#     rng = random.Random(); rng.seed(1); return rng.sample(dataset, size)
# Bound to a constant so the case selection within each partition is
# bit-stable across runs and matches the paper's exact 300/300/150 picks.
_PAPER_SAMPLE_SEED = 1


def load_hotpotqa_paper_split(
    partition: str,
    *,
    max_cases: int,
    config: str = "fullwiki",
    cache_dir: str | None = None,
) -> list[Case]:
    """Load one of the GEPA paper's 150/300/300 splits, bit-identical to the artifact.

    Reproduces the artifact's data pipeline from
    :class:`gepa_artifact.benchmarks.benchmark.Benchmark`:

    1. Load HF ``hotpot_qa`` with the ``fullwiki`` config (paper-matching
       — both ``fullwiki`` and ``distractor`` configs share the same 90k
       train questions, only the per-case ``context`` paragraph shape
       differs, and we retrieve from the wiki dump anyway).
    2. Take HotpotQA's ``train`` split as the *source* (the artifact's
       ``Benchmark`` derives all three of its splits from the train pool
       — *not* from HotpotQA's dev set).
    3. Slice the source by the artifact's fixed fractions:
       ``test = [0, 40%)``, ``validation = [40%, 80%)``, ``train = [80%, 100%)``.
    4. Subsample with ``random.Random(1).sample(slice, max_cases)`` —
       the artifact's fixed sampling seed.

    ``partition`` must be one of ``"train"`` / ``"validation"`` / ``"test"``.
    ``max_cases`` matches the artifact's per-partition defaults
    (``150`` train, ``300`` val, ``300`` test) but can be reduced for
    quicker dry runs.

    To keep the ``--retrieval distractor`` opt-out working offline, the
    ``config`` arg is exposed; pass ``"distractor"`` to load the same
    questions with per-case distractor paragraphs attached.
    """
    if partition not in _PAPER_PARTITION_BOUNDS:
        raise ValueError(
            f"load_hotpotqa_paper_split partition must be one of {sorted(_PAPER_PARTITION_BOUNDS)}, got {partition!r}."
        )
    if max_cases < 1:
        raise ValueError(f"load_hotpotqa_paper_split max_cases must be >= 1, got {max_cases}.")

    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Loading HotpotQA from HuggingFace requires the optional `datasets` dependency. "
            "Install it with `pip install datasets`."
        ) from exc

    dataset = load_dataset(
        "hotpot_qa",
        config,
        split="train",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    total = len(dataset)
    start_frac, end_frac = _PAPER_PARTITION_BOUNDS[partition]
    start = int(start_frac * total)
    end = int(end_frac * total)
    slice_size = end - start
    if slice_size <= 0:
        raise RuntimeError(
            f"HotpotQA train split too small for the {partition} partition: total={total}, slice_size={slice_size}."
        )

    take = min(max_cases, slice_size)
    # ``random.Random(seed).sample(range(n), k)`` picks the same *positions*
    # as ``random.Random(seed).sample(list_of_items, k)`` would pick
    # *items* — so sampling local indices then mapping back to global
    # row IDs reproduces the artifact's selection without materializing
    # the 36k-row slice as a Python list.
    import random as _random

    local_indices = _random.Random(_PAPER_SAMPLE_SEED).sample(range(slice_size), take)
    global_indices = [start + i for i in local_indices]
    records: list[Mapping[str, Any]] = [dataset[i] for i in global_indices]
    return cases_from_records(records)
