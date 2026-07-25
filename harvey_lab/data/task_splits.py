"""Deterministic practice-area-level splitters for Harvey LAB.

LAB tasks are grouped into practice areas (25 in the full benchmark:
contracts, corporate-ma, tax, litigation, …). Splitting at the
*practice-area* level — rather than the task level — keeps held-out areas
disjoint from the rest, so an evaluation on a held-out pool measures
cross-domain behavior (a leakage-free measure).

* :func:`fixed_val_split` — a fixed cross-area validation pool used by
  ``evaluate --split validation``.
* :func:`stratified_case_cap` — round-robin cap that keeps the area set as
  wide as possible (``evaluate --test-size``).

All helpers are deterministic given their ``(records, …, seed)`` inputs and
group by each record's ``practice_area``.
"""

from __future__ import annotations

import random as _random
from collections.abc import Sequence

from .dataset import HarveyLabRecord


__all__ = [
    "fixed_val_split",
    "stratified_case_cap",
]


def _group_by_area(records: Sequence[HarveyLabRecord]) -> dict[str, list[HarveyLabRecord]]:
    by_area: dict[str, list[HarveyLabRecord]] = {}
    for record in records:
        by_area.setdefault(record.practice_area, []).append(record)
    return by_area


def _round_robin_take(groups: list[list[HarveyLabRecord]], n: int) -> list[HarveyLabRecord]:
    """Take ``n`` items round-robin across ``groups`` (stratified, stable)."""
    out: list[HarveyLabRecord] = []
    idx = 0
    while len(out) < n:
        progressed = False
        for g in groups:
            if idx < len(g):
                out.append(g[idx])
                progressed = True
                if len(out) >= n:
                    return out
        if not progressed:
            break
        idx += 1
    return out


def fixed_val_split(
    cases: Sequence[HarveyLabRecord],
    *,
    n_val_areas: int,
    val_size: int | None,
    seed: int = 0,
) -> tuple[list[HarveyLabRecord], list[HarveyLabRecord], set[str]]:
    """Carve a FIXED cross-practice-area validation pool from the case set.

    The validation set depends ONLY on ``(cases, n_val_areas, val_size,
    seed)`` — never on the train size — so it stays constant across a
    train-size sweep. ``n_val_areas`` WHOLE practice areas are chosen
    deterministically as validation (disjoint from the train pool);
    ``val_size`` optionally caps the val case count (stratified across the
    val areas). Returns ``(train_pool_cases, val_cases, val_area_ids)``.
    Degenerate pools: 0 areas → ([], [], set()); 1 area → that area is BOTH
    train pool and val (leaky last resort, not a crash).
    """
    by_area = _group_by_area(cases)
    areas = sorted(by_area)
    n_areas = len(areas)
    if n_areas == 0:
        return [], [], set()
    if n_areas == 1:
        only = list(by_area[areas[0]])
        return only, list(only), set(areas)

    n_val = max(1, min(int(n_val_areas), n_areas - 1))
    shuffled = list(areas)
    _random.Random(seed).shuffle(shuffled)
    val_area_ids = set(shuffled[:n_val])

    train_pool: list[HarveyLabRecord] = []
    val_pool_by_area: list[list[HarveyLabRecord]] = []
    for area in areas:  # stable sorted order
        if area in val_area_ids:
            val_pool_by_area.append(by_area[area])
        else:
            train_pool.extend(by_area[area])

    val_cases: list[HarveyLabRecord] = [c for group in val_pool_by_area for c in group]
    if val_size is not None and val_size > 0 and len(val_cases) > val_size:
        val_cases = _round_robin_take(val_pool_by_area, val_size)

    assert train_pool, "fixed_val_split left no train-pool cases"
    assert val_cases, "fixed_val_split left no validation cases"
    return train_pool, val_cases, val_area_ids


def stratified_case_cap(
    train_pool: Sequence[HarveyLabRecord],
    n: int | None,
    *,
    seed: int = 0,
) -> list[HarveyLabRecord]:
    """Cap the pool to ``n`` cases, round-robin across practice areas.

    Keeps the practice-area set as wide as possible at every ``n`` — the
    only varying axis is per-area depth. ``n=None`` returns the full pool
    unchanged. ``seed`` shuffles the within-area order deterministically, so
    the cases chosen to represent each area vary across seeds while world
    coverage and the train-size nesting property are preserved.
    """
    pool = list(train_pool)
    if n is None or n >= len(pool):
        return pool
    by_area = _group_by_area(pool)
    rng = _random.Random(seed)
    groups: list[list[HarveyLabRecord]] = []
    for area in sorted(by_area):
        group = by_area[area]
        rng.shuffle(group)
        groups.append(group)
    return _round_robin_take(groups, n)
