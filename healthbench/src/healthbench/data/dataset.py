"""HealthBench loading, caching, and the frozen train/test split.

The data is OpenAI's public HealthBench release (MIT, via
``openai/simple-evals``): one JSONL row per conversation, each with its own
physician-written rubric. Three subsets ship:

* ``full`` — all 5,000 conversations.
* ``hard`` — the 1,000 the paper found hardest for frontier models.
* ``consensus`` — 3,671 conversations graded only on the 34 consensus
  criteria multiple physicians agreed on.

Splits are derived, not stored: a conversation is in ``test`` when the first
byte of ``sha256(prompt_id)`` is below ``TEST_FRACTION * 256``, else
``train``. Within a split rows are ordered by that hash, so any ``--limit``
prefix is a uniform random sample and the same prompt always lands in the same
split, across subsets and file updates.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx


_BLOB = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench"
SUBSETS: dict[str, str] = {
    "full": f"{_BLOB}/2025-05-07-06-14-12_oss_eval.jsonl",
    "hard": f"{_BLOB}/hard_2025-05-08-21-00-10.jsonl",
    "consensus": f"{_BLOB}/consensus_2025-05-09-20-00-46.jsonl",
}
TEST_FRACTION = 0.3
Split = Literal["train", "test", "all"]


@dataclass(frozen=True)
class RubricItem:
    criterion: str
    points: float
    tags: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        # Matches simple-evals' rendering (raw JSON points, so "[10] ..."), which the judge prompt was tuned on.
        return f"[{self.points}] {self.criterion}"


@dataclass(frozen=True)
class HealthBenchRecord:
    prompt_id: str
    # The conversation so far, ending in a user turn: ``[{"role", "content"}]``.
    prompt: list[dict[str, str]]
    rubrics: list[RubricItem]
    example_tags: list[str] = field(default_factory=list)


def default_cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(root) / "healthbench"


def record_from_raw(raw: dict[str, Any]) -> HealthBenchRecord:
    return HealthBenchRecord(
        prompt_id=raw["prompt_id"],
        prompt=[{"role": m["role"], "content": m["content"]} for m in raw["prompt"]],
        rubrics=[RubricItem(r["criterion"], r["points"], list(r.get("tags", []))) for r in raw["rubrics"]],
        example_tags=list(raw.get("example_tags", [])),
    )


def fetch_subset(subset: str, cache_dir: Path | None = None) -> Path:
    """Download ``subset``'s JSONL into the cache (once) and return its path."""
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}; expected one of {sorted(SUBSETS)}.")
    url = SUBSETS[subset]
    path = (cache_dir or default_cache_dir()) / url.rsplit("/", 1)[1]
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
    tmp.rename(path)
    return path


def _bucket(prompt_id: str) -> bytes:
    return hashlib.sha256(prompt_id.encode()).digest()


def split_records(records: list[HealthBenchRecord], split: Split) -> list[HealthBenchRecord]:
    """Filter ``records`` to ``split`` and order them by hash (see module doc)."""
    cutoff = int(TEST_FRACTION * 256)
    if split == "test":
        chosen = [r for r in records if _bucket(r.prompt_id)[0] < cutoff]
    elif split == "train":
        chosen = [r for r in records if _bucket(r.prompt_id)[0] >= cutoff]
    else:
        chosen = list(records)
    return sorted(chosen, key=lambda r: _bucket(r.prompt_id))


def load_records(
    subset: str = "full",
    split: Split = "test",
    *,
    limit: int | None = None,
    cache_dir: Path | None = None,
) -> list[HealthBenchRecord]:
    path = fetch_subset(subset, cache_dir)
    with path.open() as f:
        records = [record_from_raw(json.loads(line)) for line in f if line.strip()]
    chosen = split_records(records, split)
    return chosen[:limit] if limit is not None else chosen
