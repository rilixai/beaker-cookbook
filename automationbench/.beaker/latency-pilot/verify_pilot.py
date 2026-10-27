"""Validate the frozen H20 pilot without importing candidate code or calling providers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def verify(root: Path = ROOT) -> dict[str, object]:
    provenance = json.loads((root / "provenance.json").read_text())
    project = root / "project"
    expected_paths = set()
    for entry in provenance["targets"]:
        path = Path(entry["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe target path: {path}")
        expected_paths.add(path.as_posix())
        actual = hashlib.sha256((project / path).read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise ValueError(f"H20 candidate bytes changed: {path}")
    actual_paths = {
        path.relative_to(project).as_posix()
        for tree in ("src", "skills", "prompts")
        for path in (project / tree).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if actual_paths != expected_paths:
        raise ValueError("H20 candidate tree contains missing or additional files")
    for path, digest in provenance["dependency_sha256"].items():
        if hashlib.sha256((project / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Pinned dependency file changed: {path}")
    for path, digest in provenance["trusted_policy_sha256"].items():
        if hashlib.sha256((project / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Trusted pilot policy changed: {path}")
    ids_by_split = {}
    for split, expected_count in (("train", 36), ("test", 18)):
        data = (root / "dataset" / f"{split}.jsonl").read_bytes()
        manifest = provenance["dataset"][split]
        if hashlib.sha256(data).hexdigest() != manifest["pilot_sha256"]:
            raise ValueError(f"Frozen {split} dataset changed")
        rows = [json.loads(line) for line in data.splitlines()]
        if len(rows) != expected_count or [row["id"] for row in rows] != manifest["case_ids"]:
            raise ValueError(f"Wrong {split} membership")
        counts: dict[str, int] = {}
        for row in rows:
            metadata = row["metadata"]
            if metadata["source_split"] != split:
                raise ValueError("Pilot case moved between source splits")
            domain = metadata["domain"]
            counts[domain] = counts.get(domain, 0) + 1
        if len(counts) != 6 or set(counts.values()) != {manifest["per_domain"]}:
            raise ValueError(f"Wrong domain balance in {split}")
        ids_by_split[split] = set(manifest["case_ids"])
    if ids_by_split["train"] & ids_by_split["test"]:
        raise ValueError("TRAIN and TEST overlap")
    return {
        "source_run_id": provenance["source_run_id"],
        "candidate": provenance["selected_hypothesis"],
        "candidate_files": len(expected_paths),
        "train_cases": 36,
        "test_cases": 18,
        "model": provenance["model"],
        "reasoning_effort": provenance["reasoning_effort"],
        "max_concurrency": provenance["max_concurrency"],
        "status": "verified; no evaluations launched",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
