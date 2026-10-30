"""Fetch checksum-pinned upstream data and keep the official test set held out."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jev_banking77.agent import LABELS


REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
BASE_URL = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{REVISION}/banking_data"
SHA256 = {
    "categories.json": "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}
VALIDATION_PER_LABEL = 20
SPLIT_SEED = "jev-banking77-v1"
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
            raise ValueError("Unknown BANKING77 intent")
        return value


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    input: Message
    expected: Expected


def source_file(name: str, cache_dir: Path) -> bytes:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / name
    if path.exists():
        content = path.read_bytes()
    else:
        with urlopen(f"{BASE_URL}/{name}", timeout=60) as response:
            content = response.read()
    if hashlib.sha256(content).hexdigest() != SHA256[name]:
        raise ValueError(f"Checksum mismatch for {name}; no data was prepared.")
    path.write_bytes(content)
    return content


def _order(row: Row) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}:{row.id}".encode()).hexdigest()


def make_splits(train: bytes, test: bytes) -> dict[str, list[Row]]:
    sources: dict[str, list[Row]] = {}
    for split, content in (("train", train), ("test", test)):
        records = csv.DictReader(io.StringIO(content.decode("utf-8")))
        sources[split] = [
            Row(
                id=f"banking77-{split}-{index:05d}",
                input=Message(text=r["text"]),
                expected=Expected(intent=r["category"]),
            )
            for index, r in enumerate(records)
        ]
    if len(sources["train"]) != 10003 or len(sources["test"]) != 3080:
        raise ValueError("Unexpected upstream dataset size")
    if {r.input.text for r in sources["train"]} & {r.input.text for r in sources["test"]}:
        raise ValueError("Official train/test texts overlap")
    grouped: dict[str, list[Row]] = defaultdict(list)
    for row in sources["train"]:
        grouped[row.expected.intent].append(row)
    if set(grouped) != set(LABELS):
        raise ValueError("Training data must cover every intent")
    splits: dict[str, list[Row]] = {"train": [], "validation": [], "test": sources["test"]}
    for rows in grouped.values():
        rows.sort(key=_order)
        splits["validation"].extend(rows[:VALIDATION_PER_LABEL])
        splits["train"].extend(rows[VALIDATION_PER_LABEL:])
    for rows in splits.values():
        rows.sort(key=_order)
    return splits


def prepare(output_dir: Path, cache_dir: Path) -> dict[str, int]:
    if json.loads(source_file("categories.json", cache_dir)) != list(LABELS):
        raise ValueError("Packaged intent IDs differ from upstream")
    splits = make_splits(source_file("train.csv", cache_dir), source_file("test.csv", cache_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        path = output_dir / SPLIT_FILES[split]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(row.model_dump_json() + "\n" for row in rows))
    counts = {name: len(rows) for name, rows in splits.items()}
    manifest = {
        "upstream_revision": REVISION,
        "sha256": SHA256,
        "split_seed": SPLIT_SEED,
        "counts": counts,
        "files": SPLIT_FILES,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return counts


def load_rows(path: Path) -> list[Row]:
    rows = [Row.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({row.id for row in rows}) != len(rows):
        raise ValueError("Dataset must be non-empty with unique row IDs")
    return rows
