"""Build the balanced PI demo locally. This command never uploads data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from automationbench_skills.data.make_splits import _allocate_largest_remainder
from automationbench_skills.data.tasks import PUBLIC_DOMAINS, Sample, load_split, read_split_names, task_family


BENCHMARK_COMMIT = "858718b19e5d66c74451432783d34ee64a550a15"
DATASET_NAME = "automationbench-pi-demo210"
SEED = 0
PER_DOMAIN = {"train": 20, "test": 10}


def select_demo(split: str) -> list[Sample]:
    """Equal domain quotas, proportional family quotas, no split reassignment."""
    per_domain = PER_DOMAIN[split]
    by_domain: dict[str, list[Sample]] = defaultdict(list)
    for sample in load_split(split):
        by_domain[sample.domain].append(sample)
    selected: list[Sample] = []
    for domain in PUBLIC_DOMAINS:
        candidates = by_domain[domain]
        if len(candidates) < per_domain:
            raise ValueError(f"{split}/{domain} needs {per_domain} eligible tasks, but has {len(candidates)}")
        families: dict[str, list[str]] = defaultdict(list)
        for sample in candidates:
            families[task_family(sample.task_name)].append(sample.task_name)
        quotas = _allocate_largest_remainder({family: len(names) for family, names in families.items()}, per_domain)
        # Reset for each domain so changes in another domain cannot perturb this one.
        rng = random.Random(SEED)
        chosen: set[str] = set()
        for family in sorted(families):
            names = sorted(families[family])
            rng.shuffle(names)
            chosen.update(names[: quotas[family]])
        selected.extend(sample for sample in candidates if sample.task_name in chosen)
    return selected


def demo_rows(split: str) -> list[dict[str, Any]]:
    return [
        {
            "id": sample.task_name,
            "input": {
                "task_name": sample.task_name,
                "prompt": "\n\n".join(
                    str(message.get("content") or "") for message in sample.prompt if message.get("role") == "user"
                ).strip(),
            },
            "expected": {"assertions": sample.info["assertions"]},
            "metadata": {
                "domain": sample.domain,
                "task_family": task_family(sample.task_name),
                "source_split": split,
                "benchmark_commit": BENCHMARK_COMMIT,
                "selection_seed": SEED,
                "selection_method": "domain-balanced-family-stratified",
            },
            "group_key": sample.domain,
        }
        for sample in select_demo(split)
    ]


def write_demo(output_dir: Path) -> dict[str, Any]:
    """Write two JSONL files and a count/hash manifest; refuse to replace files."""
    rows = {split: demo_rows(split) for split in PER_DOMAIN}
    ids = {split: {row["id"] for row in values} for split, values in rows.items()}
    excluded = set(read_split_names("excluded_cases"))
    if ids["train"] & ids["test"] or (ids["train"] | ids["test"]) & excluded:
        raise ValueError("Demo contains overlapping splits or excluded tasks")
    for split in PER_DOMAIN:
        if len(rows[split]) != len(ids[split]):
            raise ValueError(f"Duplicate task IDs in {split}")
    paths = [output_dir / name for name in ("train.jsonl", "test.jsonl", "verification.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError(f"Demo files already exist in {output_dir}; choose a new output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "dataset_name": DATASET_NAME,
        "benchmark_commit": BENCHMARK_COMMIT,
        "seed": SEED,
        "selection": "Equal domain quotas; largest-remainder family quotas within each eligible source split.",
        "excluded_case_count": len(excluded),
        "selected_excluded_cases": [],
        "train_test_overlap": [],
        "case_count": sum(len(values) for values in rows.values()),
        "split_counts": {split: len(values) for split, values in rows.items()},
        "per_domain": {},
        "per_family": {},
        "sha256": {},
    }
    for split, values in rows.items():
        content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in values).encode()
        (output_dir / f"{split}.jsonl").write_bytes(content)
        report["sha256"][f"{split}.jsonl"] = hashlib.sha256(content).hexdigest()
        report["per_domain"][split] = dict(Counter(row["metadata"]["domain"] for row in values))
        report["per_family"][split] = {
            domain: dict(
                Counter(row["metadata"]["task_family"] for row in values if row["metadata"]["domain"] == domain)
            )
            for domain in PUBLIC_DOMAINS
        }
    (output_dir / "verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="local directory for the JSONL files and manifest"
    )
    args = parser.parse_args()
    report = write_demo(args.output_dir)
    print(
        json.dumps(
            {"output_dir": str(args.output_dir.resolve()), "split_counts": report["split_counts"], "uploaded": False}
        )
    )


if __name__ == "__main__":
    main()
