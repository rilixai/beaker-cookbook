"""Fetch the checksum-pinned LiveNewsBench release and write question/answer rows.

Only the question, the gold answer and a little metadata leave this module. The
upstream rows also carry the source articles the questions were written from;
those are dropped so no answer evidence reaches the agent or the uploaded dataset.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from pydantic import BaseModel, ConfigDict, Field


REVISION = "8a6b96e7904893977ac8a3ae3342ae3d39f89c2e"
RELEASE = "jan_2026_release_2"
BASE_URL = f"https://raw.githubusercontent.com/YunfanZhang42/LiveNewsBench/{REVISION}/datasets/{RELEASE}"
SHA256 = {
    "train.jsonl": "c86b2a177e3d2115a12985272d4cb9bddf66fcadb3d9553f4058965ceaaf8634",
    "val.jsonl": "f51db4d9552e34b334ec395f731bb4bf866157776ca526489ab7e3e2cabe3abf",
    "test.jsonl": "c55adb924eb4aa5e634f8ff624d706d300e925382234545a0873bb7b3ac113b3",
    "human_verified_test.jsonl": "e733b10302678c9d57d5e19cf41b735ddb852e81ba418436f2a9da28c96425e6",
}
EXPECTED_COUNTS = {"train": 612, "val": 171, "test": 346, "human_verified_test": 200}
# Beaker optimizes on `train` and reports on `test` (upstream val). Upstream test,
# which contains every human-verified question, stays out of Beaker entirely.
SPLIT_FILES = {
    "train": "optimization/train.jsonl",
    "val": "optimization/test.jsonl",
    "test": "heldout/test.jsonl",
}
SPLITS = (*SPLIT_FILES, "human_verified_test")


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1)


class Expected(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1)


class Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_split: str
    event_date: str
    category: str
    human_verified: bool


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    input: Question
    expected: Expected
    metadata: Metadata


def source_file(name: str, cache_dir: Path) -> bytes:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / name
    if path.exists():
        content = path.read_bytes()
    else:
        with urlopen(f"{BASE_URL}/{name}", timeout=120) as response:
            content = response.read()
    if hashlib.sha256(content).hexdigest() != SHA256[name]:
        raise ValueError(f"Checksum mismatch for {name}; no data was prepared.")
    path.write_bytes(content)
    return content


def row_id(question: str) -> str:
    return "lnb-" + hashlib.sha256(question.encode()).hexdigest()[:12]


def make_rows(sources: dict[str, bytes]) -> dict[str, list[Row]]:
    records = {
        split: [json.loads(line) for line in sources[f"{split}.jsonl"].decode().splitlines() if line.strip()]
        for split in EXPECTED_COUNTS
    }
    if {split: len(rows) for split, rows in records.items()} != EXPECTED_COUNTS:
        raise ValueError("Unexpected upstream split sizes")
    verified = {record["question"] for record in records["human_verified_test"]}
    if not verified <= {record["question"] for record in records["test"]}:
        raise ValueError("Upstream human-verified questions are no longer a subset of test")
    splits: dict[str, list[Row]] = {}
    for split in SPLIT_FILES:
        splits[split] = [
            Row(
                id=row_id(record["question"]),
                input=Question(question=record["question"]),
                expected=Expected(answer=record["answer"]),
                metadata=Metadata(
                    source_split=split,
                    event_date=record["date"],
                    category=record["category"],
                    human_verified=record["question"] in verified,
                ),
            )
            for record in records[split]
        ]
    questions = [row.input.question for rows in splits.values() for row in rows]
    if len(set(questions)) != len(questions):
        raise ValueError("Upstream splits share or repeat questions")
    return splits


def prepare(output_dir: Path, cache_dir: Path) -> dict[str, int]:
    splits = make_rows({name: source_file(name, cache_dir) for name in SHA256})
    for split, rows in splits.items():
        path = output_dir / SPLIT_FILES[split]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(row.model_dump_json() + "\n" for row in rows))
    counts = {split: len(rows) for split, rows in splits.items()}
    counts["human_verified_test"] = sum(row.metadata.human_verified for row in splits["test"])
    manifest = {
        "upstream_revision": REVISION,
        "release": RELEASE,
        "sha256": SHA256,
        "counts": counts,
        "files": SPLIT_FILES,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return counts


def load_rows(data_dir: Path, split: str) -> list[Row]:
    path = data_dir / SPLIT_FILES["test" if split == "human_verified_test" else split]
    rows = [Row.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
    if split == "human_verified_test":
        rows = [row for row in rows if row.metadata.human_verified]
    if not rows or len({row.id for row in rows}) != len(rows):
        raise ValueError("Dataset must be non-empty with unique row IDs")
    return rows
