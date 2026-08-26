"""Run-once generator for the frozen splits (the txt files are the source of truth).

75 train / 25 test per public domain (450/150 total). Within each domain the 25
held-out tasks are stratified across task families (``task_family``: the first
underscore token of the task name) by largest-remainder allocation over each
family's share of the domain, with a fixed-seed shuffle inside each family.
``simple.txt`` lists the 200 unscored ``simple``-domain tasks as optional
extra *training* material only.

Regenerate with ``uv run python -m automationbench_skills.data.make_splits``
(only meaningful when the pinned automation-bench dependency changes).
"""

from __future__ import annotations

import random
from collections import defaultdict

from automationbench_skills.data.tasks import (
    PUBLIC_DOMAINS,
    Sample,
    load_samples,
    split_path,
    task_family,
)


SEED = 0
TEST_PER_DOMAIN = 25


def _allocate_largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    """Allocate `total` slots across families proportionally to `counts`."""
    domain_size = sum(counts.values())
    quotas = {f: total * c / domain_size for f, c in counts.items()}
    alloc = {f: int(q) for f, q in quotas.items()}
    remainder = total - sum(alloc.values())
    # Break remainder ties deterministically: larger fraction first, then name.
    order = sorted(counts, key=lambda f: (-(quotas[f] - alloc[f]), f))
    for f in order[:remainder]:
        alloc[f] += 1
    # A family cannot give more tasks than it has.
    for f in order:
        while alloc[f] > counts[f]:
            alloc[f] -= 1
            for g in order:
                if alloc[g] < counts[g] and g != f:
                    alloc[g] += 1
                    break
    return alloc


def make_splits() -> tuple[list[str], list[str], list[str]]:
    samples = load_samples(include_simple=True)
    by_domain: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_domain[s.domain].append(s)

    train: list[str] = []
    test: list[str] = []
    for domain in PUBLIC_DOMAINS:
        domain_samples = by_domain[domain]
        families: dict[str, list[str]] = defaultdict(list)
        for s in domain_samples:
            families[task_family(s.task_name)].append(s.task_name)
        alloc = _allocate_largest_remainder({f: len(v) for f, v in families.items()}, TEST_PER_DOMAIN)
        rng = random.Random(SEED)
        domain_test: set[str] = set()
        for family in sorted(families):
            names = sorted(families[family])
            rng.shuffle(names)
            domain_test.update(names[: alloc[family]])
        # Keep the benchmark's own task order inside each split file.
        test.extend(s.task_name for s in domain_samples if s.task_name in domain_test)
        train.extend(s.task_name for s in domain_samples if s.task_name not in domain_test)

    simple = [s.task_name for s in by_domain["simple"]]
    return train, test, simple


def main() -> None:
    train, test, simple = make_splits()
    for name, names in (("train", train), ("test", test), ("simple", simple)):
        path = split_path(name)
        path.write_text("".join(f"{n}\n" for n in names))
        print(f"wrote {path} ({len(names)} tasks)")


if __name__ == "__main__":
    main()
