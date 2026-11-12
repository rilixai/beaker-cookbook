"""Fetch checksum-pinned upstream data and keep the official test set held out."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

from pydantic import BaseModel, ConfigDict, Field, field_validator

from clinc150_router.contracts import LABELS, OUT_OF_SCOPE


REVISION = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
SOURCE_URL = f"https://raw.githubusercontent.com/clinc/oos-eval/{REVISION}/data/data_full.json"
SOURCE_NAME = "data_full.json"
SHA256 = "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"
SPLIT_SEED = "clinc150-router-v1"
# Our split name -> (upstream in-scope key, upstream out-of-scope key, rows per intent, out-of-scope rows).
UPSTREAM = {
    "train": ("train", "oos_train", 100, 100),
    "validation": ("val", "oos_val", 20, 100),
    "test": ("test", "oos_test", 30, 1000),
}
SPLIT_FILES = {
    "train": "optimization/train.jsonl",
    "validation": "optimization/test.jsonl",
    "test": "heldout/test.jsonl",
}


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)


class Expected(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: str

    @field_validator("intent")
    @classmethod
    def known_label(cls, value: str) -> str:
        if value not in LABELS:
            raise ValueError("Unknown CLINC150 intent")
        return value


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    input: Message
    expected: Expected


def source_file(cache_dir: Path) -> bytes:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / SOURCE_NAME
    if path.exists():
        content = path.read_bytes()
    else:
        with urlopen(SOURCE_URL, timeout=60) as response:
            content = response.read()
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError(f"Checksum mismatch for {SOURCE_NAME}; no data was prepared.")
    path.write_bytes(content)
    return content


def _order(row: Row) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}:{row.id}".encode()).hexdigest()


def make_splits(source: bytes) -> dict[str, list[Row]]:
    upstream = json.loads(source)
    splits: dict[str, list[Row]] = {}
    for split, (in_scope_key, oos_key, per_intent, oos_count) in UPSTREAM.items():
        pairs = [*upstream[in_scope_key], *upstream[oos_key]]
        rows = [
            Row(id=f"clinc150-{split}-{index:05d}", input=Message(text=text), expected=Expected(intent=label))
            for index, (text, label) in enumerate(pairs)
        ]
        counts = Counter(row.expected.intent for row in rows)
        expected_counts = {label: per_intent for label in LABELS if label != OUT_OF_SCOPE} | {OUT_OF_SCOPE: oos_count}
        if counts != expected_counts:
            raise ValueError(f"Unexpected upstream {split} size or label set")
        # Any prefix of the sorted file is a fair sample.
        splits[split] = sorted(rows, key=_order)
    return splits


def duplicate_texts_across_splits(splits: dict[str, list[Row]]) -> dict[str, int]:
    """Upstream repeats a few utterances across splits; record them instead of silently editing the benchmark."""
    texts = {name: {row.input.text for row in rows} for name, rows in splits.items()}
    names = list(texts)
    return {f"{a}/{b}": len(texts[a] & texts[b]) for i, a in enumerate(names) for b in names[i + 1 :]}


def prepare(output_dir: Path, cache_dir: Path) -> dict[str, int]:
    splits = make_splits(source_file(cache_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        path = output_dir / SPLIT_FILES[split]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(row.model_dump_json() + "\n" for row in rows))
    counts = {name: len(rows) for name, rows in splits.items()}
    manifest = {
        "upstream_url": SOURCE_URL,
        "upstream_revision": REVISION,
        "sha256": {SOURCE_NAME: SHA256},
        "split_seed": SPLIT_SEED,
        "counts": counts,
        "out_of_scope_counts": {
            name: sum(row.expected.intent == OUT_OF_SCOPE for row in rows) for name, rows in splits.items()
        },
        "upstream_duplicate_texts_across_splits": duplicate_texts_across_splits(splits),
        "files": SPLIT_FILES,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return counts


def load_rows(path: Path) -> list[Row]:
    rows = [Row.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({row.id for row in rows}) != len(rows):
        raise ValueError("Dataset must be non-empty with unique row IDs")
    return rows
