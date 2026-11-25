"""Regenerate DecisionPivot's core track from the pinned upstream generators and write contrast-group splits.

Nothing here edits or copies the generators. ``prepare`` fetches the pinned
upstream commit into a cache, runs ``generators/regenerate.py --check`` there as
an unmodified subprocess (published seed, then seeds 1 and 2), and checks every
items file against a pinned SHA-256: release 1.1's for the published draws, and
the hashes recorded below for the fresh draws.

Splits (one row = one contrast group; a group never crosses files):
  - The seven families whose generator takes a seed: train = seed 1,
    validation = seed 2, held out = the published release 1.1 items.
  - The other six: the published groups, split by a seeded hash within each
    family and level, about 60% train, 20% validation, 20% held out.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


UPSTREAM_REPO = "https://github.com/StartLuxLabs/DecisionPivot.git"
UPSTREAM_COMMIT = "fd57688f033864e64c452561ffdfcba9739d1811"
DEFAULT_CACHE = Path.home() / ".cache/decisionpivot" / f"upstream-{UPSTREAM_COMMIT[:12]}"
RELEASE = "1.1"
PUBLISHED_SEED = 20261007
TRAIN_SEED = 1
VALIDATION_SEED = 2
SPLIT_SEED = "decisionpivot-v1"
HELDOUT_FRACTION = 0.2
VALIDATION_FRACTION = 0.2
ITEMS_PER_FAMILY = 600

# family -> layer, in upstream's order.
FAMILIES: dict[str, str] = {
    "answerable_facts": "A",
    "perm_inherit": "B",
    "policy_case": "B",
    "log_count_check": "C",
    "sched_conflict": "D",
    "state_goal": "D",
    "toolcall_valid": "D",
    "triage_order": "D",
    "clause_change": "E",
    "dep_resolve": "E",
    "ledger_reconcile": "F",
    "record_rule": "F",
    "filter_match": "G",
}
SEEDED = (
    "clause_change",
    "dep_resolve",
    "ledger_reconcile",
    "perm_inherit",
    "record_rule",
    "sched_conflict",
    "triage_order",
)
UNSEEDED = tuple(f for f in FAMILIES if f not in SEEDED)

# SHA-256 of items.jsonl for the fresh draws, recorded with CPython 3.12.12 from
# the pinned generators. Release 1.1 hashes come from upstream releases/v1.1/index.json.
FRESH_SHA256: dict[int, dict[str, str]] = {
    TRAIN_SEED: {
        "clause_change": "c8d9a2092854a5cc50efbc21cbe1523a25dd76a528cb5cc1c3d02bf478ebdb9f",
        "dep_resolve": "d418799f8a9b48f8e470fe1d933915d97302502f8986b4c69bb63f7d296c3c31",
        "ledger_reconcile": "427fb19b97f2a9d0c4cb3ff0587f0794203fa59293ce85b8cf6256064195a1da",
        "perm_inherit": "f38898530d22efbd52fb24f87ec853d0993518bfbfca1fa8afaa7a9f30421305",
        "record_rule": "c62183e24fbf9928c7f5463e92dc5640761e4c9e6a5e9d7d1740cabdc3c3be72",
        "sched_conflict": "d4b42ba3ae9e7c61a0c4434a540f60ad3ff04bae9c1e671247ea2e53e04396bd",
        "triage_order": "37adda3783ab01ad4eba8def0ceee3336f82b99ce0a4db1b70713a2d403b1190",
    },
    VALIDATION_SEED: {
        "clause_change": "5d2673df076481066164838d7bb9ee895384921239afdf10db079db2c57fd82a",
        "dep_resolve": "462e12ff91f61f2cbededd93adaf86916346f94b4faffdd5c0d69ca2a4c86bec",
        "ledger_reconcile": "da9c25b770b67aeb9c76e91e9cbd49efb595e2fec0a85fefd3551749a3f93a34",
        "perm_inherit": "d28dcdee24e2084041eea79eb73a40dec94ddd77928e89dd990d6bd4c1870035",
        "record_rule": "0dd5d64aaaaab1da6ef7fc0c724d3de5912ee1f70b329591fa7b326463941cf8",
        "sched_conflict": "815ed4f2f7272a3b0a58b2916ecfe806107721c0ac580b46cbc543610d98527e",
        "triage_order": "82f09236149b60aa18cc79972526b4023ac2f8c92e1016a9c08948c6633e0999",
    },
}

Split = Literal["train", "validation", "heldout"]
SPLIT_FILES: dict[str, str] = {
    "train": "optimization/train.jsonl",
    "validation": "optimization/test.jsonl",
    "heldout": "heldout/test.jsonl",
}


class Question(BaseModel):
    """One typed question, exactly as upstream freezes it: type, instructions, criteria (option key -> text)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["choice"]
    instructions: str = Field(min_length=1)
    criteria: dict[str, str] = Field(min_length=2)

    @property
    def keys(self) -> list[str]:
        return list(self.criteria)


class Item(BaseModel):
    """What the agent may see of one item: its id, state and question. Never gold, level, deviation or spec."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    state: str | dict[str, JsonValue]
    questions: dict[Literal["q"], Question]

    @property
    def question(self) -> Question:
        return self.questions["q"]


class CaseInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    family: str
    items: tuple[Item, ...] = Field(min_length=2, max_length=4)


class Expected(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gold: dict[str, str]
    level: int = Field(ge=1, le=3)
    layer: str


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^dp-")
    input: CaseInput
    expected: Expected

    @model_validator(mode="after")
    def gold_matches_items(self) -> Row:
        ids = [item.id for item in self.input.items]
        if len(set(ids)) != len(ids) or set(self.expected.gold) != set(ids):
            raise ValueError("gold must name every item of the group exactly once")
        for item in self.input.items:
            if self.expected.gold[item.id] not in item.question.criteria:
                raise ValueError(f"gold of {item.id} is not one of its options")
        return self

    @property
    def group(self) -> str:
        return self.id.removeprefix("dp-")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(args: Sequence[str], cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def fetch_upstream(cache_dir: Path = DEFAULT_CACHE) -> Path:
    """The pinned upstream commit, checked out in the cache (git verifies the content by the commit hash)."""
    if not (cache_dir / ".git").is_dir():
        cache_dir.mkdir(parents=True, exist_ok=True)
        _git(["init", "-q"], cache_dir)
        _git(["fetch", "-q", "--depth", "1", UPSTREAM_REPO, UPSTREAM_COMMIT], cache_dir)
        _git(["-c", "advice.detachedHead=false", "checkout", "-q", UPSTREAM_COMMIT], cache_dir)
    head = _git(["rev-parse", "HEAD"], cache_dir)
    if head != UPSTREAM_COMMIT:
        raise ValueError(f"Upstream cache {cache_dir} is at {head}, not {UPSTREAM_COMMIT}")
    # generated/ is upstream's own output folder and is gitignored; anything else changed means an edited copy.
    if _git(["status", "--porcelain"], cache_dir):
        raise ValueError(f"Upstream cache {cache_dir} has local changes; delete it and prepare again.")
    return cache_dir


def regenerate(upstream: Path, *, python: str = sys.executable) -> str:
    """Run upstream regenerate.py --check, unmodified: published seed for all 13 families, then seeds 1 and 2."""
    logs = []
    for extra in ([], ["--seed", str(TRAIN_SEED), "--seed", str(VALIDATION_SEED)]):
        result = subprocess.run(
            [python, "-I", "generators/regenerate.py", *extra, "--check"],
            cwd=upstream,
            capture_output=True,
            text=True,
        )
        logs.append(result.stdout + result.stderr)
        if result.returncode != 0:
            raise ValueError(f"Upstream regeneration failed; no data was prepared.\n{logs[-1]}")
    return "\n".join(logs)


def items_path(upstream: Path, family: str, seed: int | None) -> Path:
    root = upstream / "generated" if seed is None else upstream / "generated" / f"seed_{seed}"
    return root / "tasks" / family / "items.jsonl"


def release_hashes(upstream: Path) -> dict[str, str]:
    index = json.loads((upstream / "releases/v1.1/index.json").read_text())["tasks"]
    return {family: index[family]["items_sha256"] for family in FAMILIES}


def read_items(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"Hash mismatch for {path}: {actual} != {expected_sha256}; no data was prepared.")
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    if len(items) != ITEMS_PER_FAMILY:
        raise ValueError(f"{path} has {len(items)} items, expected {ITEMS_PER_FAMILY}")
    return items


def to_rows(family: str, items: Iterable[Mapping[str, Any]]) -> list[Row]:
    """Upstream items to one row per contrast group. Only id, state and questions go into ``input``."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        if item.get("gold") is None or not item.get("group"):
            raise ValueError(f"{item.get('id')}: core items must have a gold option and a group")
        groups[str(item["group"])].append(item)
    rows = []
    for group, members in groups.items():
        levels = {m["meta"]["level"] for m in members}
        if len(levels) != 1:
            raise ValueError(f"Group {group} mixes levels {sorted(levels)}")
        rows.append(
            Row(
                id=f"dp-{group}",
                input=CaseInput(
                    family=family,
                    items=tuple(
                        Item.model_validate({"id": m["id"], "state": m["state"], "questions": m["questions"]})
                        for m in members
                    ),
                ),
                expected=Expected(
                    gold={m["id"]: str(m["gold"]) for m in members},
                    level=levels.pop(),
                    layer=FAMILIES[family],
                ),
            )
        )
    return rows


def _hash(purpose: str, key: str) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}:{purpose}:{key}".encode()).hexdigest()


def split_published(rows: Sequence[Row]) -> dict[str, list[Row]]:
    """Seeded hash split of published groups, stratified by family and level (20% held out, 20% validation)."""
    strata: dict[tuple[str, int], list[Row]] = defaultdict(list)
    for row in rows:
        strata[(row.input.family, row.expected.level)].append(row)
    out: dict[str, list[Row]] = {"train": [], "validation": [], "heldout": []}
    for members in strata.values():
        members = sorted(members, key=lambda r: _hash("split", r.id))
        heldout = math.floor(len(members) * HELDOUT_FRACTION + 0.5)
        validation = math.floor(len(members) * VALIDATION_FRACTION + 0.5)
        out["heldout"].extend(members[:heldout])
        out["validation"].extend(members[heldout : heldout + validation])
        out["train"].extend(members[heldout + validation :])
    return out


def make_splits(
    published: Mapping[str, Sequence[Row]], fresh: Mapping[int, Mapping[str, Sequence[Row]]]
) -> dict[str, list[Row]]:
    """Every family's rows into train / validation / heldout; each file ordered by a seeded hash."""
    splits: dict[str, list[Row]] = {"train": [], "validation": [], "heldout": []}
    for family in FAMILIES:
        if family in SEEDED:
            splits["train"].extend(fresh[TRAIN_SEED][family])
            splits["validation"].extend(fresh[VALIDATION_SEED][family])
            splits["heldout"].extend(published[family])
        else:
            for name, rows in split_published(published[family]).items():
                splits[name].extend(rows)
    for rows in splits.values():
        rows.sort(key=lambda r: _hash("order", r.id))
    return splits


def counts(rows: Sequence[Row]) -> dict[str, Any]:
    return {
        "groups": len(rows),
        "items": sum(len(r.input.items) for r in rows),
        "groups_by_family": dict(sorted(Counter(r.input.family for r in rows).items())),
        "items_by_family_level": {
            family: dict(
                sorted(
                    Counter(
                        str(r.expected.level) for r in rows if r.input.family == family for _ in r.input.items
                    ).items()
                )
            )
            for family in FAMILIES
        },
    }


def load_all(upstream: Path) -> tuple[dict[str, list[Row]], dict[int, dict[str, list[Row]]], dict[str, Any]]:
    """Read and hash-check every generated items file; rows per family for the published and the fresh draws."""
    published_hashes = release_hashes(upstream)
    published = {
        family: to_rows(family, read_items(items_path(upstream, family, None), published_hashes[family]))
        for family in FAMILIES
    }
    fresh = {
        seed: {
            family: to_rows(family, read_items(items_path(upstream, family, seed), FRESH_SHA256[seed][family]))
            for family in SEEDED
        }
        for seed in (TRAIN_SEED, VALIDATION_SEED)
    }
    hashes = {"published": published_hashes, **{f"seed_{s}": FRESH_SHA256[s] for s in FRESH_SHA256}}
    return published, fresh, hashes


def first_per_family(rows: Sequence[Row], n: int) -> list[Row]:
    """The first ``n`` groups of each family, in file order (a seeded hash, so a fair sample)."""
    taken: Counter[str] = Counter()
    kept = []
    for row in rows:
        if taken[row.input.family] < n:
            taken[row.input.family] += 1
            kept.append(row)
    return kept


def prepare(
    output_dir: Path,
    cache_dir: Path = DEFAULT_CACHE,
    *,
    python: str = sys.executable,
    groups_per_family: int | None = None,
) -> dict[str, Any]:
    upstream = fetch_upstream(cache_dir)
    regenerate(upstream, python=python)
    published, fresh, hashes = load_all(upstream)
    splits = make_splits(published, fresh)
    if groups_per_family is not None:  # a small dataset for cheap trial runs
        splits = {name: first_per_family(rows, groups_per_family) for name, rows in splits.items()}
    seen: set[str] = set()
    for rows in splits.values():
        ids = {item.id for r in rows for item in r.input.items}
        if ids & seen:
            raise ValueError("An item appears in two splits")
        seen |= ids
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = output_dir / SPLIT_FILES[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(r.model_dump_json() + "\n" for r in rows))
    manifest: dict[str, Any] = {
        "benchmark": "DecisionPivot core track",
        "upstream": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT,
        "release": RELEASE,
        "python": platform.python_version(),
        "seeds": {"published": PUBLISHED_SEED, "train": TRAIN_SEED, "validation": VALIDATION_SEED},
        "split_seed": SPLIT_SEED,
        "groups_per_family": groups_per_family,
        "seeded_families": list(SEEDED),
        "unseeded_families": list(UNSEEDED),
        "split_rule": {
            "seeded_families": {"train": "seed 1", "validation": "seed 2", "heldout": "published release 1.1"},
            "unseeded_families": {
                "pool": "published release 1.1",
                "fractions": {"train": 0.6, "validation": VALIDATION_FRACTION, "heldout": HELDOUT_FRACTION},
                "stratified_by": ["family", "level"],
                "unit": "contrast group",
            },
        },
        "items_sha256": hashes,
        "files": SPLIT_FILES,
        "counts": {name: counts(rows) for name, rows in splits.items()},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {name: {"groups": len(rows), "items": manifest["counts"][name]["items"]} for name, rows in splits.items()}


def load_rows(path: Path) -> list[Row]:
    rows = [Row.model_validate_json(line) for line in path.read_text().split("\n") if line.strip()]
    if not rows or len({row.id for row in rows}) != len(rows):
        raise ValueError("Dataset must be non-empty with unique row IDs")
    return rows


LEAK_WORDS = 12
_WORD = re.compile(r"[^\W_]+(?:['’.-][^\W_]+)*")


def _strings(value: JsonValue) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def find_leaks(rows: Iterable[Row], files: Mapping[str, str], *, words: int = LEAK_WORDS) -> list[dict[str, str]]:
    """Item states, or runs of ``words``+ consecutive words from a state, that appear in the given texts.

    Words are compared case-insensitively, ignoring punctuation and whitespace.
    A structured state is read as the text of its string values, in order.
    """
    windows: dict[tuple[str, ...], str] = {}
    texts = {name: " ".join(_words(text)) for name, text in files.items()}
    for name, text in files.items():
        tokens = _words(text)
        for i in range(len(tokens) - words + 1):
            windows.setdefault(tuple(tokens[i : i + words]), name)
    leaks: list[dict[str, str]] = []
    for row in rows:
        for item in row.input.items:
            tokens = [w for text in _strings(item.state) for w in _words(text)]
            if len(tokens) < words:
                joined = " ".join(tokens)
                hit = next((n for n, t in texts.items() if joined and f" {joined} " in f" {t} "), None)
                if hit:
                    leaks.append({"item": item.id, "file": hit, "text": joined})
                continue
            for i in range(len(tokens) - words + 1):
                hit = windows.get(tuple(tokens[i : i + words]))
                if hit:
                    leaks.append({"item": item.id, "file": hit, "text": " ".join(tokens[i : i + words])})
                    break
    return leaks
