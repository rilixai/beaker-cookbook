"""Build a temporary Beaker dataset from AutomationBench's frozen task splits."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from automationbench_skills import Sample, load_split


def _round_robin(samples: list[Sample]) -> list[Sample]:
    by_domain: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_domain[sample.domain].append(sample)
    domains = sorted(by_domain)
    ordered: list[Sample] = []
    index = 0
    while any(index < len(by_domain[domain]) for domain in domains):
        ordered.extend(by_domain[domain][index] for domain in domains if index < len(by_domain[domain]))
        index += 1
    return ordered


def _row(sample: Sample, source_split: str) -> dict[str, object]:
    return {
        "id": sample.task_name,
        "input": {"task_name": sample.task_name},
        "expected": {"partial_credit": 1.0},
        "metadata": {"domain": sample.domain, "source_split": source_split},
        "group_key": sample.domain,
    }


def write_dataset(output_dir: Path, *, train_count: int, val_count: int, test_count: int) -> dict[str, int]:
    train_pool = _round_robin(load_split("train"))
    test_pool = _round_robin(load_split("test"))
    if train_count + val_count > len(train_pool) or test_count > len(test_pool):
        raise ValueError("requested subset exceeds the frozen AutomationBench splits")
    selected = {
        "train": train_pool[:train_count],
        "val": train_pool[train_count : train_count + val_count],
        "test": test_pool[:test_count],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split, samples in selected.items():
        counts[split] = len(samples)
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as output:
            for sample in samples:
                source_split = "test" if split == "test" else "train"
                output.write(json.dumps(_row(sample, source_split), sort_keys=True) + "\n")
    return counts
