"""Frozen train/test split of ``officeqa_pro.csv`` and its deterministic generator.

This split is ours. It is **not** a published partition and is not
comparable to any number in the OfficeQA Pro report. See ``README.md`` in
this directory for the procedure.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from officeqa.data.dataset import EvalRecord


SEED = 0
N_TEST = 40

Stratum = tuple[str, int]  # (n_source_files bucket, decade)


def n_files_bucket(n: int) -> str:
    return "1" if n == 1 else "2" if n == 2 else "3+"


def stratum_of(record: EvalRecord) -> Stratum:
    return (n_files_bucket(record.n_source_files), record.earliest_year // 10 * 10)


def largest_remainder(sizes: Sequence[int], total: int) -> list[int]:
    """Apportion ``total`` slots across groups proportionally to ``sizes``.

    Each group gets ``floor(size * total / sum)``; leftover slots go to the
    groups with the largest fractional remainders (ties broken by input
    order, then by size). No group receives more slots than its size.
    """
    n = sum(sizes)
    if total > n:
        raise ValueError("total exceeds population")
    quotas = [s * total / n for s in sizes]
    alloc = [int(q) for q in quotas]
    remainder = total - sum(alloc)
    order = sorted(range(len(sizes)), key=lambda i: (-(quotas[i] - alloc[i]), -sizes[i], i))
    for i in order:
        if remainder == 0:
            break
        if alloc[i] < sizes[i]:
            alloc[i] += 1
            remainder -= 1
    return alloc


@dataclass(frozen=True)
class SplitResult:
    train: tuple[str, ...]
    test: tuple[str, ...]
    strata: dict[Stratum, tuple[int, int]]  # stratum -> (n_train, n_test)


def _round_robin(groups: Iterable[Sequence[str]]) -> list[str]:
    queues = [list(g) for g in groups if g]
    out: list[str] = []
    while queues:
        for q in list(queues):
            out.append(q.pop(0))
            if not q:
                queues.remove(q)
    return out


def generate_split(records: Sequence[EvalRecord], *, n_test: int = N_TEST, seed: int = SEED) -> SplitResult:
    """Stratified split; see the module docstring / ``README.md`` for the procedure."""
    by_stratum: dict[Stratum, list[str]] = defaultdict(list)
    for r in sorted(records, key=lambda r: r.uid):
        by_stratum[stratum_of(r)].append(r.uid)
    strata = sorted(by_stratum)  # deterministic order: ("1", 1930), ("1", 1940), ...

    rng = random.Random(seed)
    for s in strata:
        rng.shuffle(by_stratum[s])

    test_alloc = largest_remainder([len(by_stratum[s]) for s in strata], n_test)

    test_groups: list[list[str]] = []
    train_groups: list[list[str]] = []
    counts: dict[Stratum, tuple[int, int]] = {}
    for s, k in zip(strata, test_alloc, strict=True):
        ids = by_stratum[s]
        test_groups.append(ids[:k])
        train_groups.append(ids[k:])
        counts[s] = (len(ids) - k, k)

    return SplitResult(train=tuple(_round_robin(train_groups)), test=tuple(_round_robin(test_groups)), strata=counts)


def write_split(result: SplitResult, out_dir: Path) -> None:
    (out_dir / "train.txt").write_text("\n".join(result.train) + "\n", encoding="utf-8")
    (out_dir / "test.txt").write_text("\n".join(result.test) + "\n", encoding="utf-8")


def strata_table(result: SplitResult) -> str:
    lines = ["| files | decade | train | test | total |", "|---|---|---|---|---|"]
    for (bucket, decade), (tr, te) in sorted(result.strata.items()):
        lines.append(f"| {bucket} | {decade}s | {tr} | {te} | {tr + te} |")
    lines.append(
        f"| **all** | | **{len(result.train)}** | **{len(result.test)}** | **{len(result.train) + len(result.test)}** |"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    """``python -m officeqa.splits [--csv path]`` — regenerate ``train.txt`` / ``test.txt``."""
    import argparse

    from officeqa.data.dataset import load_records, load_records_from_csv

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=None, help="local officeqa_pro.csv (default: download pinned revision)")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args(argv)

    records = load_records_from_csv(args.csv) if args.csv else load_records()
    result = generate_split(records)
    write_split(result, args.out_dir)
    print(strata_table(result))
    print(f"\nwrote {len(result.train)} train / {len(result.test)} test ids to {args.out_dir}")
