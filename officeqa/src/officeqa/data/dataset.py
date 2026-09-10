"""OfficeQA Pro question loading and the ground-truth / agent-input boundary.

Two record types, deliberately separate:

* :class:`EvalRecord` — the full CSV row, including ``answer``,
  ``source_docs`` and ``source_files``. Used **only** by the scorer and by
  split generation.
* :class:`AgentInput` — what the agent is allowed to see: ``uid``,
  ``question`` and a neutral corpus manifest. It is a frozen dataclass with
  ``slots`` and no other fields, so one cannot be constructed holding an
  answer or a source field.

``datasets`` is imported lazily so this module stays importable / testable
offline; tests monkeypatch :func:`_download_rows`.
"""

from __future__ import annotations

import csv
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from officeqa import config
from officeqa.data.manifest import CorpusManifest


logger = logging.getLogger(__name__)


# ``source_files`` entries look like ``treasury_bulletin_1941_01.txt``. In the
# shipped CSV multi-file rows are newline-separated (the spec expected ``;``);
# accept both.
_SOURCE_FILE_SEP = re.compile(r"[;\n]")
_BULLETIN_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})")


@dataclass(frozen=True, slots=True)
class AgentInput:
    """The agent-visible sample. Carries **only** uid, question, corpus.

    ``slots=True`` + ``frozen=True`` means no attribute other than these
    three can ever be attached to an instance; ground truth cannot ride
    along by accident.
    """

    uid: str
    question: str
    corpus: CorpusManifest

    def to_json(self) -> dict[str, Any]:
        return {"uid": self.uid, "question": self.question, "corpus": self.corpus.to_json()}


@dataclass(frozen=True, slots=True)
class EvalRecord:
    """Full ``officeqa_pro.csv`` row. Never handed to the agent."""

    uid: str
    question: str
    answer: str
    source_docs: str
    source_files: str
    difficulty: str

    @property
    def source_file_list(self) -> tuple[str, ...]:
        return tuple(s.strip() for s in _SOURCE_FILE_SEP.split(self.source_files) if s.strip())

    @property
    def source_basenames(self) -> tuple[str, ...]:
        """``treasury_bulletin_YYYY_MM`` stems, one per referenced bulletin."""
        return tuple(Path(f).stem for f in self.source_file_list)

    @property
    def n_source_files(self) -> int:
        return len(self.source_file_list)

    @property
    def earliest_year(self) -> int:
        years = [int(m.group(1)) for m in _BULLETIN_RE.finditer(self.source_files)]
        if not years:
            raise ValueError(f"{self.uid}: no treasury_bulletin_YYYY_MM in source_files={self.source_files!r}")
        return min(years)

    def agent_input(self, corpus: CorpusManifest) -> AgentInput:
        return AgentInput(uid=self.uid, question=self.question, corpus=corpus)


def _record_from_row(row: Mapping[str, Any]) -> EvalRecord:
    return EvalRecord(
        uid=str(row["uid"]).strip(),
        question=str(row["question"]),
        answer=str(row["answer"]),
        source_docs=str(row.get("source_docs") or ""),
        source_files=str(row.get("source_files") or ""),
        difficulty=str(row.get("difficulty") or ""),
    )


def _download_rows(data_file: str, revision: str) -> Iterable[Mapping[str, Any]]:
    """Fetch one CSV of the gated HF dataset. Needs ``HF_TOKEN``."""
    from datasets import load_dataset

    ds = load_dataset(
        config.OFFICEQA_HF_REPO,
        data_files=data_file,
        split="train",
        revision=revision,
    )
    return list(ds)


def load_records(
    *,
    data_file: str = config.OFFICEQA_PRO_FILE,
    revision: str = config.OFFICEQA_DATASET_REVISION,
) -> list[EvalRecord]:
    """Load all rows of ``data_file`` at the pinned revision, sorted by uid."""
    records = [_record_from_row(r) for r in _download_rows(data_file, revision)]
    records.sort(key=lambda r: r.uid)
    uids = [r.uid for r in records]
    if len(set(uids)) != len(uids):
        raise ValueError("duplicate uids in dataset")
    return records


def load_records_from_csv(path: Path) -> list[EvalRecord]:
    """Offline loader for a local CSV (tests, split regeneration)."""
    with path.open(newline="", encoding="utf-8") as fh:
        records = [_record_from_row(r) for r in csv.DictReader(fh)]
    records.sort(key=lambda r: r.uid)
    return records


# --- Splits -----------------------------------------------------------------

SPLITS_DIR = Path(__file__).resolve().parent.parent / "splits"
KNOWN_SPLITS: tuple[str, ...] = ("train", "test", "dev")


def read_split_ids(split: str) -> list[str]:
    """Frozen uid list for ``train`` / ``test`` (order is round-robin over strata)."""
    if split not in ("train", "test"):
        raise ValueError(f"unknown split {split!r}; known: {KNOWN_SPLITS}")
    path = SPLITS_DIR / f"{split}.txt"
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [i for i in ids if i and not i.startswith("#")]


def select_split(records: Sequence[EvalRecord], split: str) -> list[EvalRecord]:
    """Order ``records`` per the frozen split file; error on missing uids."""
    by_uid = {r.uid: r for r in records}
    ids = read_split_ids(split)
    missing = [i for i in ids if i not in by_uid]
    if missing:
        raise ValueError(f"split {split!r} references uids absent from the dataset: {missing[:5]}...")
    return [by_uid[i] for i in ids]


def load_split(
    split: str,
    *,
    revision: str = config.OFFICEQA_DATASET_REVISION,
    limit: int | None = None,
) -> list[EvalRecord]:
    """Load the ``EvalRecord``s for a split, in frozen file order.

    ``train`` / ``test`` come from ``officeqa_pro.csv`` and the committed
    id lists. ``dev`` is the 113 ``easy`` rows of ``officeqa_full.csv`` —
    smoke-test material only: "easy" was defined upstream as "two frontier
    agents already solved it", so it saturates fast and is not a useful
    diagnostic for a strong agent.
    """
    if split == "dev":
        full = load_records(data_file=config.OFFICEQA_FULL_FILE, revision=revision)
        records = [r for r in full if r.difficulty == "easy"]
    else:
        records = select_split(load_records(revision=revision), split)
    if limit is not None:
        records = records[:limit]
    return records
