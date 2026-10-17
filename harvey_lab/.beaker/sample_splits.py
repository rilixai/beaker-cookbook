"""Sample training tasks uniformly within quotas derived from the fixed test set."""

from __future__ import annotations

import random
from collections import Counter, defaultdict


def practice_area(task_id: str) -> str:
    return task_id.split("/", 1)[0]


def validate_splits(train: list[str], test: list[str]) -> None:
    if not train or not test:
        raise ValueError("train and test must both contain tasks")
    if len(set(train)) != len(train) or len(set(test)) != len(test):
        raise ValueError("frozen splits contain duplicate task IDs")
    if set(train) & set(test):
        raise ValueError("frozen train and test splits overlap")


def sample_train(train: list[str], test: list[str], count: int, seed: int) -> list[str]:
    """Use largest-remainder quotas, then sample without replacement in each area.

    Each quota differs from count * test_area_share by less than one task.
    Equal remainders are resolved by area name. Insufficient capacity fails
    instead of redistributing a quota and changing the target distribution.
    """
    validate_splits(train, test)
    if not 1 <= count <= len(train):
        raise ValueError(f"train size must be between 1 and {len(train)}")
    target = Counter(map(practice_area, test))
    quotas = {area: count * size // len(test) for area, size in target.items()}
    remainder_order = sorted(target, key=lambda area: (-(count * target[area] % len(test)), area))
    for area in remainder_order[: count - sum(quotas.values())]:
        quotas[area] += 1
    candidates: dict[str, list[str]] = defaultdict(list)
    for task_id in train:
        candidates[practice_area(task_id)].append(task_id)
    rng = random.Random(seed)
    selected: list[str] = []
    for area in sorted(quotas):
        quota = quotas[area]
        if quota > len(candidates[area]):
            raise ValueError(
                f"{area}: need {quota} train tasks to match test, but only {len(candidates[area])} are available; "
                "reduce --train-limit"
            )
        selected.extend(rng.sample(sorted(candidates[area]), quota))
    rng.shuffle(selected)
    return selected


def distribution_report(train: list[str], test: list[str]) -> dict[str, dict[str, int | float]]:
    train_counts = Counter(map(practice_area, train))
    test_counts = Counter(map(practice_area, test))
    return {
        area: {
            "train_count": train_counts[area],
            "test_count": test_counts[area],
            "train_share": train_counts[area] / len(train),
            "test_share": test_counts[area] / len(test),
        }
        for area in sorted(train_counts.keys() | test_counts.keys())
    }
