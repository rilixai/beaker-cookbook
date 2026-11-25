"""Fetch the checksum-pinned MMDM release and write reproducible, stratified splits.

MMDM publishes a single test set, so this recipe makes its own: about 60% train,
20% validation (``optimization/test.jsonl``) and 20% held out
(``heldout/test.jsonl``, final evaluation only).

Questions with more options than the decision API accepts in one request are
dropped before splitting. The request guard requires every original option in
every request, so those questions could never be answered. The manifest lists
them.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal
from urllib.request import urlopen

from pydantic import BaseModel, ConfigDict, Field, model_validator


DATASET = "shenjunhao/mmdm"
REVISION = "1551e88c3b5290d6c5d3fb4c01b0ec00c57cfe7e"
FILENAME = "mmdm.jsonl"
SOURCE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{FILENAME}"
SHA256 = "70fb1ba6a61406bbb25b3125e63c21f3e1fc2ac0dce5b57525ff1cbc0c61d2ed"
UPSTREAM_ROWS = 5448
# The largest catalog the decision API takes in one request (upstream --option-cap 255).
OPTION_CAP = 255
SPLIT_SEED = "mmdm-v1"
HELDOUT_FRACTION = 0.2
VALIDATION_FRACTION = 0.2
SPLIT_FILES = {
    "train": "optimization/train.jsonl",
    "validation": "optimization/test.jsonl",
    "test": "heldout/test.jsonl",
}
CATEGORIES = (
    "agent_state_tracking",
    "answer_preference",
    "code_database",
    "constraint_planning",
    "deductive_reasoning",
    "emotion_recognition",
    "evidence_validation",
    "intent_routing",
    "option_logic",
    "policy_exception",
    "quantitative_temporal_reasoning",
    "reading_comprehension",
    "score_rubric",
    "tool_skill_selection",
    "workflow_decision",
    "world_knowledge",
)
AGENT_CATEGORIES = (
    "agent_state_tracking",
    "tool_skill_selection",
    "workflow_decision",
    "intent_routing",
    "policy_exception",
    "evidence_validation",
    "constraint_planning",
)

Task = Literal["choice", "multichoice", "noul", "score"]
LabelSemantics = Literal["categorical_hard", "set_hard", "categorical_soft", "stochastic_ordinal_score"]


class Option(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    criteria: str | None = None


class Question(BaseModel):
    """Everything the agent may see: one MMDM request, without labels or category."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: str
    question: str
    options: tuple[Option, ...] = Field(min_length=2)
    task: Task
    # int | float keeps upstream's integer score values byte-identical in requests.
    score_values: tuple[int | float, ...] | None = None

    @model_validator(mode="after")
    def consistent(self) -> Question:
        ids = [o.id for o in self.options]
        if len(set(ids)) != len(ids):
            raise ValueError("Option IDs must be unique")
        if (self.task == "score") != (self.score_values is not None):
            raise ValueError("score_values are required for score tasks and only for them")
        if self.score_values is not None and len(self.score_values) != len(self.options):
            raise ValueError("score_values must align with options")
        return self

    @property
    def option_ids(self) -> list[str]:
        return [o.id for o in self.options]


class Expected(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label_semantics: LabelSemantics
    probabilities: tuple[float, ...]
    correct_option_ids: tuple[str, ...]
    task_category: str
    task_subcategory: str
    source: str


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^mmdm-\d+$")
    input: Question
    expected: Expected

    @model_validator(mode="after")
    def labels_match_options(self) -> Row:
        if len(self.expected.probabilities) != len(self.input.options):
            raise ValueError("Label probabilities must align with options")
        if not set(self.expected.correct_option_ids) <= set(self.input.option_ids):
            raise ValueError("Correct option IDs must be option IDs")
        if not math.isclose(math.fsum(self.expected.probabilities), 1.0, abs_tol=1e-6):
            raise ValueError("Label probabilities must sum to one")
        return self


def source_file(cache_dir: Path) -> bytes:
    """The pinned release, from cache when present; any checksum mismatch fails closed."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / FILENAME
    if path.exists():
        content = path.read_bytes()
    else:
        with urlopen(SOURCE_URL, timeout=300) as response:
            content = response.read()
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError(f"Checksum mismatch for {FILENAME}; no data was prepared.")
    path.write_bytes(content)
    return content


def convert(records: Iterable[Mapping[str, Any]]) -> list[Row]:
    """Upstream records to rows. ``gold`` and ``answer.probabilities`` must agree on every row."""
    rows: list[Row] = []
    disagreements: list[Any] = []
    for record in records:
        answer = record.get("answer")
        if not isinstance(answer, Mapping) or answer.get("probabilities") != record["gold"]:
            disagreements.append(record.get("id"))
            continue
        rows.append(
            Row(
                id=f"mmdm-{record['id']}",
                input=Question(
                    state=record["state"],
                    question=record["question"],
                    options=tuple(Option(**o) for o in record["options"]),
                    task=record["task"],
                    score_values=record["score_values"],
                ),
                expected=Expected(
                    label_semantics=record["label_semantics"],
                    probabilities=tuple(record["gold"]),
                    correct_option_ids=tuple(record["correct_option_ids"]),
                    task_category=record["task_category"],
                    task_subcategory=record["task_subcategory"],
                    source=record["source"],
                ),
            )
        )
    if disagreements:
        raise ValueError(
            f"gold and answer.probabilities disagree on {len(disagreements)} rows "
            f"(first: {disagreements[:5]}); no data was prepared."
        )
    if len({row.id for row in rows}) != len(rows):
        raise ValueError("Upstream question IDs must be unique")
    return rows


def _hash(purpose: str, row_id: str) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}:{purpose}:{row_id}".encode()).hexdigest()


def make_splits(rows: Sequence[Row]) -> dict[str, list[Row]]:
    """Deterministic hash split, stratified by task_category x label_semantics.

    Each stratum is ordered by a seeded hash; the first 20% (rounded half up)
    are held out, the next 20% are validation, the rest train. Each output is
    then ordered by a second seeded hash so any prefix is a fair sample.
    Assignment depends only on a row's stratum and ID, so a category subset
    gets exactly the rows the full dataset assigns it.
    """
    strata: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for row in rows:
        strata[(row.expected.task_category, row.expected.label_semantics)].append(row)
    splits: dict[str, list[Row]] = {"train": [], "validation": [], "test": []}
    for members in strata.values():
        members.sort(key=lambda r: _hash("split", r.id))
        heldout = math.floor(len(members) * HELDOUT_FRACTION + 0.5)
        validation = math.floor(len(members) * VALIDATION_FRACTION + 0.5)
        splits["test"].extend(members[:heldout])
        splits["validation"].extend(members[heldout : heldout + validation])
        splits["train"].extend(members[heldout + validation :])
    for members in splits.values():
        members.sort(key=lambda r: _hash("order", r.id))
    return splits


def parse_categories(value: str | None) -> tuple[str, ...]:
    if not value:
        return CATEGORIES
    selected = tuple(dict.fromkeys(c.strip() for c in value.split(",") if c.strip()))
    unknown = sorted(set(selected) - set(CATEGORIES))
    if not selected or unknown:
        raise ValueError(f"Unknown MMDM categories: {unknown or value!r}")
    return selected


def prepare(output_dir: Path, cache_dir: Path, categories: Sequence[str] = CATEGORIES) -> dict[str, Any]:
    content = source_file(cache_dir)
    # split on "\n" only: states contain U+2028 and other characters str.splitlines() breaks on
    rows = convert(json.loads(line) for line in content.decode("utf-8").split("\n") if line.strip())
    if len(rows) != UPSTREAM_ROWS:
        raise ValueError(f"Expected {UPSTREAM_ROWS} upstream rows, found {len(rows)}")
    over_cap = [row for row in rows if len(row.input.options) > OPTION_CAP]
    kept = [row for row in rows if len(row.input.options) <= OPTION_CAP]
    splits = {
        name: [row for row in members if row.expected.task_category in categories]
        for name, members in make_splits(kept).items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, members in splits.items():
        path = output_dir / SPLIT_FILES[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(row.model_dump_json() + "\n" for row in members))
    dropped = [row for row in over_cap if row.expected.task_category in categories]
    manifest: dict[str, Any] = {
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "source_file": FILENAME,
        "sha256": SHA256,
        "split_seed": SPLIT_SEED,
        "fractions": {"train": 0.6, "validation": VALIDATION_FRACTION, "test": HELDOUT_FRACTION},
        "stratified_by": ["task_category", "label_semantics"],
        "categories": list(categories),
        "files": SPLIT_FILES,
        "counts": {name: len(members) for name, members in splits.items()},
        "counts_by_category": {
            name: dict(sorted(Counter(r.expected.task_category for r in members).items()))
            for name, members in splits.items()
        },
        "dropped_over_option_cap": {
            "option_cap": OPTION_CAP,
            "reason": "More options than one decision request accepts; every request must carry every option.",
            "count": len(dropped),
            "by_category": dict(sorted(Counter(r.expected.task_category for r in dropped).items())),
            "ids": [row.id for row in dropped],
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "counts": manifest["counts"],
        "dropped_over_option_cap": len(dropped),
        "categories": len(categories),
    }


def load_rows(path: Path) -> list[Row]:
    rows = [Row.model_validate_json(line) for line in path.read_text().split("\n") if line.strip()]
    if not rows or len({row.id for row in rows}) != len(rows):
        raise ValueError("Dataset must be non-empty with unique row IDs")
    return rows
