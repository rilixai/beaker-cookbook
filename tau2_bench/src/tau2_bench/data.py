"""Fetch checksum-pinned upstream data, list tasks, and build reproducible splits."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal
from urllib.request import urlopen

from pydantic import BaseModel, ConfigDict, Field

from tau2_bench import UPSTREAM_COMMIT


Domain = Literal["airline", "retail"]
DOMAINS: tuple[Domain, ...] = ("airline", "retail")
BASE_URL = f"https://raw.githubusercontent.com/sierra-research/tau2-bench/{UPSTREAM_COMMIT}/data"
# Every file an English, text-mode airline or retail run reads at this commit.
SHA256 = {
    "tau2/domains/airline/db.json": "1af9fea6e03ca7ca15a22bb3fcaf3e351393e3fc9070b6777947da8996f7531b",
    "tau2/domains/airline/policy.md": "10dc0525421521208be39cee235bba84a16e2bcba9899eb93d92cd81d2f62fc4",
    "tau2/domains/airline/split_tasks.json": "b22ced4d9a9850ac9aea31c53bdcb6d6009058140bd9acc7db37c1d36222ba8b",
    "tau2/domains/airline/tasks.json": "ccd8ba737b4cc371415af70151187788f728d6108d0916e73bb4317b40542052",
    "tau2/domains/retail/db.json": "413a65160adbdb5fde0ffc0015c49b6d70250b10c18128de169b597af7766765",
    "tau2/domains/retail/policy.md": "2c9652afbce57d6e087768d37cda64d31c53d50b3e3225cfdb791bac66466467",
    "tau2/domains/retail/split_tasks.json": "ed0580ec52575b63fbf76568af42490da6ee7783ecb4aa81af46961291358f20",
    "tau2/domains/retail/tasks.json": "8e03ebce7901bd6218e7a7dc3105faa9324091a68058f7fe61c65262868812e8",
    "tau2/multilingual/en/airline_metadata_en.json": "d40cb3ab048cb0818483da9721bbdfb365c193c9cf4b8771094a2c6db5062b2d",
    "tau2/multilingual/en/pack.yaml": "6453ef2c2f06ca34fb0a4f90841780913a2f7af28b680a08be402190876e6435",
    "tau2/multilingual/en/retail_metadata_en.json": "357c5ad5768b30a0bdf05a126d5588238034ad206ccb39c688b439e368ce2ac9",
    "tau2/multilingual/en/retail_tasks_en.json": "431d7c1ba3554bfcf687509b0bd0c6fab9b7c04edc745dae94d509658f7b5753",
}
SPLIT_SEED = "tau2-bench-v1"
SPLIT_FILES = {
    "train": "optimization/train.jsonl",
    "validation": "optimization/test.jsonl",
    "test": "heldout/test.jsonl",
}


def data_dir() -> Path:
    return Path(os.environ["TAU2_DATA_DIR"])


def ensure_data() -> Path:
    """Download any missing upstream file and verify every file's checksum."""
    root = data_dir()
    for name, digest in SHA256.items():
        path = root / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            continue
        with urlopen(f"{BASE_URL}/{name}", timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch for {name}; no data was written.")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = NamedTemporaryFile(dir=path.parent, prefix=f"{path.name}.", suffix=".partial", delete=False)
        partial = Path(temporary.name)
        try:
            with temporary:
                temporary.write(content)
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    return root


class TaskRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    domain: Domain
    task_id: str = Field(min_length=1)


class Row(BaseModel):
    """One Beaker case. Expected outcomes stay inside upstream; rows only reference tasks."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    input: TaskRef
    expected: dict[str, object] = Field(default_factory=dict, max_length=0)


def row_for(domain: Domain, task_id: str) -> Row:
    return Row(id=f"{domain}-{task_id}", input=TaskRef(domain=domain, task_id=task_id))


def upstream_splits(domain: str) -> dict[str, list[str]]:
    """Upstream's official ``train``/``test`` task IDs for a domain."""
    raw = json.loads((ensure_data() / f"tau2/domains/{domain}/split_tasks.json").read_text())
    return {"train": [str(i) for i in raw["train"]], "test": [str(i) for i in raw["test"]]}


def _order(domain: str, task_id: str) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}:{domain}:{task_id}".encode()).hexdigest()


def make_splits(task_ids: dict[str, list[str]], official: dict[str, dict[str, list[str]]]) -> dict[str, list[Row]]:
    """Upstream test is held out; upstream train is hash-split 75/25 into train and validation."""
    splits: dict[str, list[Row]] = {"train": [], "validation": [], "test": []}
    for domain in DOMAINS:
        ids, train, test = set(task_ids[domain]), official[domain]["train"], official[domain]["test"]
        if set(train) & set(test) or set(train) | set(test) != ids or len(ids) != len(task_ids[domain]):
            raise ValueError(f"Upstream {domain} splits must partition its tasks exactly")
        ordered = sorted(train, key=lambda task_id: _order(domain, task_id))
        cut = math.ceil(len(ordered) / 4)
        splits["validation"].extend(row_for(domain, task_id) for task_id in ordered[:cut])
        splits["train"].extend(row_for(domain, task_id) for task_id in ordered[cut:])
        splits["test"].extend(row_for(domain, task_id) for task_id in test)
    for rows in splits.values():
        rows.sort(key=lambda row: _order(row.input.domain, row.input.task_id))
    return splits


def prepare(output_dir: Path) -> dict[str, dict[str, int]]:
    # Imported here so the split logic above stays importable without loading tau2.
    from tau2_bench.benchmark import task_ids

    splits = make_splits(
        {domain: task_ids(domain) for domain in DOMAINS},
        {domain: upstream_splits(domain) for domain in DOMAINS},
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        path = output_dir / SPLIT_FILES[split]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(row.model_dump_json() + "\n" for row in rows))
    counts = {
        split: {domain: sum(row.input.domain == domain for row in rows) for domain in DOMAINS} | {"total": len(rows)}
        for split, rows in splits.items()
    }
    manifest = {
        "upstream_commit": UPSTREAM_COMMIT,
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
