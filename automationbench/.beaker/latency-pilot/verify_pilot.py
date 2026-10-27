"""Validate the frozen H20 pilot without importing candidate code or calling providers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def verify(root: Path = ROOT, *, source_dataset_dir: Path | None = None) -> dict[str, object]:
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
        manifest = provenance["dataset"][split]
        case_ids = manifest["case_ids"]
        if len(case_ids) != expected_count or len(set(case_ids)) != expected_count:
            raise ValueError(f"Wrong {split} membership")
        ids_by_split[split] = set(case_ids)
        if source_dataset_dir is None:
            continue
        data = (source_dataset_dir / f"{split}.jsonl").read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest["source_sha256"]:
            raise ValueError(f"Source {split} dataset changed")
        source_rows = [json.loads(line) for line in data.splitlines()]
        if len(source_rows) != manifest["source_cases"]:
            raise ValueError(f"Wrong source {split} size")
        rows = [row for row in source_rows if row["id"] in ids_by_split[split]]
        if [row["id"] for row in rows] != case_ids:
            raise ValueError(f"Wrong {split} membership")
        pilot_data = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode()
        if hashlib.sha256(pilot_data).hexdigest() != manifest["pilot_sha256"]:
            raise ValueError(f"Frozen {split} dataset changed")
        counts: dict[str, int] = {}
        for row in rows:
            metadata = row["metadata"]
            if metadata["source_split"] != split:
                raise ValueError("Pilot case moved between source splits")
            domain = metadata["domain"]
            counts[domain] = counts.get(domain, 0) + 1
        if len(counts) != 6 or set(counts.values()) != {manifest["per_domain"]}:
            raise ValueError(f"Wrong domain balance in {split}")
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
        "dataset_rows_verified": source_dataset_dir is not None,
        "status": "verified; no evaluations launched"
        if source_dataset_dir is not None
        else "candidate and dataset references verified; dataset rows not supplied",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dataset-dir",
        type=Path,
        help="External directory holding the source revision's TRAIN and TEST JSONL files",
    )
    args = parser.parse_args()
    print(json.dumps(verify(source_dataset_dir=args.source_dataset_dir), indent=2))
