"""Deterministic world-level data splitters for APEX-Agents.

APEX-Agents tasks are clustered into worlds (10 in the
investment-banking subset). Splitting at the *world* level — rather
than the task level — keeps train and held-out worlds disjoint so a
GEPA-optimized prompt is evaluated on worlds it never saw during
optimization (the leakage-free evaluation the plan calibrates to).

This module groups the world-aware split helpers:

* :func:`fixed_val_split` — a fixed cross-world validation pool used by
  ``evaluate --split validation`` to score a candidate on whole worlds
  disjoint from the train pool.
* :func:`stratified_case_cap` — round-robin cap to ``n`` cases that
  keeps the world set as wide as possible (``evaluate --test-size``).

All helpers are deterministic given their ``(cases, …, seed)`` inputs.
"""

from __future__ import annotations

import random as _random
from collections.abc import Sequence
from typing import Any


__all__ = [
    "fixed_val_split",
    "stratified_case_cap",
]


def _group_by_world(cases: Sequence[Any]) -> dict[str, list[Any]]:
    by_world: dict[str, list[Any]] = {}
    for case in cases:
        by_world.setdefault(str(getattr(case, "group_key", "") or ""), []).append(case)
    return by_world


def fixed_val_split(
    cases: Sequence[Any],
    *,
    n_val_worlds: int,
    val_size: int | None,
    seed: int = 0,
) -> tuple[list[Any], list[Any], set[str]]:
    """Carve a FIXED cross-world validation pool from the full case set.

    The validation set depends ONLY on ``(cases, n_val_worlds, val_size,
    seed)`` — never on the train size — so it stays *constant* across a
    train-size sweep (HotpotQA-style: fix val, grow train). ``n_val_worlds``
    WHOLE worlds are chosen deterministically as validation (disjoint from
    the train pool, so candidate selection rewards cross-world transfer —
    the Fix-1 anti-overfit property); ``val_size`` optionally caps the val
    case count (stratified across the val worlds for representativeness).

    Returns ``(train_pool_cases, val_cases, val_world_ids)``. Degenerate
    pools: 0 worlds → ([], [], set()); 1 world → that world is BOTH train
    pool and val (leaky last resort, not a crash).
    """
    by_world = _group_by_world(cases)
    worlds = sorted(by_world)
    n_worlds = len(worlds)
    if n_worlds == 0:
        return [], [], set()
    if n_worlds == 1:
        only = list(by_world[worlds[0]])
        return only, list(only), set(worlds)

    n_val = max(1, min(int(n_val_worlds), n_worlds - 1))
    shuffled = list(worlds)
    _random.Random(seed).shuffle(shuffled)
    val_world_ids = set(shuffled[:n_val])

    train_pool: list[Any] = []
    val_pool_by_world: list[list[Any]] = []
    for w in worlds:  # stable sorted order
        if w in val_world_ids:
            val_pool_by_world.append(by_world[w])
        else:
            train_pool.extend(by_world[w])

    val_cases: list[Any] = [c for group in val_pool_by_world for c in group]
    if val_size is not None and val_size > 0 and len(val_cases) > val_size:
        val_cases = _round_robin_take(val_pool_by_world, val_size)

    assert train_pool, "fixed_val_split left no train-pool cases"
    assert val_cases, "fixed_val_split left no validation cases"
    return train_pool, val_cases, val_world_ids


def _round_robin_take(groups: list[list[Any]], n: int) -> list[Any]:
    """Take ``n`` items round-robin across ``groups`` (stratified, stable)."""
    out: list[Any] = []
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


def stratified_case_cap(
    train_pool: Sequence[Any],
    n: int | None,
    *,
    mode: str = "stratified",
    seed: int = 0,
) -> list[Any]:
    """Cap the train pool to ``n`` cases.

    ``mode="stratified"`` (default): round-robin across the pool's worlds
    so the world set stays as wide as possible at every ``n`` — the only
    varying axis is per-world depth, giving a clean "more data, same
    worlds" scaling curve. ``n=None`` returns the full pool unchanged.

    ``seed`` shuffles the *within-world* case order (deterministically per
    seed, like the other splitters in this module) so the cases chosen to
    represent each world vary across seeds rather than always being each
    world's first ``k``. World coverage and the train-size nesting property
    (a smaller ``n`` is a prefix of a larger ``n`` at the same seed) are
    preserved because the per-world order is fixed once the seed is set.
    """
    pool = list(train_pool)
    if n is None or n >= len(pool):
        return pool
    if mode != "stratified":
        raise ValueError(f"stratified_case_cap mode must be 'stratified', got {mode!r}.")
    by_world = _group_by_world(pool)
    rng = _random.Random(seed)
    groups: list[list[Any]] = []
    for world in sorted(by_world):
        group = by_world[world]  # fresh list from _group_by_world; safe to shuffle in place
        rng.shuffle(group)
        groups.append(group)
    return _round_robin_take(groups, n)
