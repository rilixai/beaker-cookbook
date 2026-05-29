"""Deterministic world-level data splitters for APEX-Agents.

APEX-Agents tasks are clustered into worlds (10 in the
investment-banking subset). Splitting at the *world* level — rather
than the task level — keeps train and held-out worlds disjoint so a
GEPA-optimized prompt is evaluated on worlds it never saw during
optimization (the leakage-free evaluation the plan calibrates to).

This module groups the world-aware split helpers:

* :func:`world_held_out_val_split` — carve inner validation out of a
  train pool by holding out whole worlds (used by both the local CLI
  and the Modal ``build_spec``).
* :func:`fixed_val_split` — a fixed cross-world validation pool that
  stays constant across a train-size sweep (local CLI).
* :func:`stratified_case_cap` — round-robin cap to ``n`` cases that
  keeps the world set as wide as possible (used everywhere).
* :func:`world_level_folds` — the k-fold partitioner behind the CLI's
  ``kfold`` command: shuffle the world ids under a seed and partition
  into ``k`` near-equal, disjoint test groups.

All helpers are deterministic given their ``(cases, …, seed)`` inputs.
"""

from __future__ import annotations

import random as _random
from collections.abc import Sequence
from typing import Any


__all__ = [
    "fixed_val_split",
    "stratified_case_cap",
    "world_held_out_val_split",
    "world_level_folds",
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
    worlds" scaling curve. ``mode="frontslice"``: legacy ``pool[:n]``
    (worlds collapse at small ``n`` — confounds the curve). ``n=None``
    returns the full pool unchanged.
    """
    pool = list(train_pool)
    if n is None or n >= len(pool):
        return pool
    if mode == "frontslice":
        return pool[:n]
    if mode != "stratified":
        raise ValueError(f"stratified_case_cap mode must be 'stratified' or 'frontslice', got {mode!r}.")
    by_world = _group_by_world(pool)
    groups = [by_world[w] for w in sorted(by_world)]
    return _round_robin_take(groups, n)


def world_level_folds(
    world_ids: Sequence[str],
    k: int = 5,
    seed: int = 0,
) -> list[tuple[list[str], list[str]]]:
    """Partition ``world_ids`` into ``k`` ``(train_world_ids, test_world_ids)`` folds.

    Deterministic given ``(world_ids, k, seed)``:

    1. De-duplicate and sort ``world_ids`` (stable starting order so
       the shuffle is reproducible regardless of input ordering).
    2. Shuffle with ``random.Random(seed)``.
    3. Partition into ``k`` contiguous test groups of near-equal size
       (sizes differ by at most one).
    4. For each fold, ``train`` = every world NOT in that fold's test
       group.

    Invariants asserted before returning:

    * Every world appears in exactly one test fold (a partition).
    * Fold sizes are balanced (max - min <= 1).

    For the 10 investment-banking worlds with ``k=5`` this yields 5
    folds of 2 test worlds / 8 train worlds.
    """
    if k < 2:
        raise ValueError(f"world_level_folds requires k >= 2, got {k}.")
    unique_sorted = sorted({str(w) for w in world_ids if str(w)})
    n = len(unique_sorted)
    if n < k:
        raise ValueError(f"world_level_folds needs at least k={k} distinct worlds, got {n} ({unique_sorted}).")

    shuffled = list(unique_sorted)
    _random.Random(seed).shuffle(shuffled)

    # Contiguous near-equal partition: the first ``n % k`` folds get
    # one extra world so sizes differ by at most one.
    base, remainder = divmod(n, k)
    folds: list[tuple[list[str], list[str]]] = []
    cursor = 0
    test_groups: list[list[str]] = []
    for fold_index in range(k):
        size = base + (1 if fold_index < remainder else 0)
        test_group = shuffled[cursor : cursor + size]
        cursor += size
        test_groups.append(test_group)

    for test_group in test_groups:
        test_set = set(test_group)
        train_group = [w for w in unique_sorted if w not in test_set]
        folds.append((train_group, sorted(test_group)))

    # Invariant: the union of test groups is an exact partition of the
    # world set (every world in exactly one test fold).
    seen: list[str] = []
    for _, test_group in folds:
        seen.extend(test_group)
    assert sorted(seen) == unique_sorted, (
        f"world_level_folds did not partition the worlds: {sorted(seen)} != {unique_sorted}"
    )
    assert len(seen) == len(set(seen)), "world_level_folds produced a world in more than one test fold"

    # Invariant: balanced fold sizes (differ by at most one).
    sizes = [len(test_group) for _, test_group in folds]
    assert max(sizes) - min(sizes) <= 1, f"world_level_folds produced unbalanced folds: {sizes}"

    return folds


def world_held_out_val_split(
    train_cases: Sequence[Any],
    *,
    n_val_worlds: int,
    seed: int = 0,
) -> tuple[list[Any], list[Any]]:
    """Split a fold's train cases into ``(inner_train, validation)`` by WORLD.

    GEPA selects candidates by validation score. If validation is a random
    slice of the *same* worlds GEPA trains on, candidate selection rewards
    in-world fit and the chosen prompt collapses on unseen worlds (exactly
    what the Law fold-0 run showed: val 0.28 → held-out 0.14). Holding out
    whole worlds for the inner validation makes GEPA select for *cross-world*
    transfer instead.

    Cases are grouped by ``group_key`` (set to ``world_id`` in
    :func:`record_to_case`). ``n_val_worlds`` whole worlds are chosen
    deterministically (``random.Random(seed)``) as validation; the rest are
    inner-train. At least one world is always left for inner-train; if the
    pool has only one world the split degenerates to (all, all) so optimize
    still has a (leaky, last-resort) signal rather than crashing.
    """
    by_world: dict[str, list[Any]] = {}
    for case in train_cases:
        by_world.setdefault(str(getattr(case, "group_key", "") or ""), []).append(case)
    worlds = sorted(by_world)
    n_worlds = len(worlds)
    if n_worlds == 0:
        return [], []
    if n_worlds == 1:
        only = by_world[worlds[0]]
        return list(only), list(only)

    n_val = max(1, min(int(n_val_worlds), n_worlds - 1))
    shuffled = list(worlds)
    _random.Random(seed).shuffle(shuffled)
    val_worlds = set(shuffled[:n_val])

    inner_train: list[Any] = []
    validation: list[Any] = []
    for w in worlds:  # stable, sorted order
        (validation if w in val_worlds else inner_train).extend(by_world[w])

    assert inner_train, "world_held_out_val_split left no inner-train cases"
    assert validation, "world_held_out_val_split left no validation cases"
    # Invariant: inner-train and validation worlds are disjoint.
    it_worlds = {str(getattr(c, "group_key", "") or "") for c in inner_train}
    assert it_worlds.isdisjoint(val_worlds), "inner-train and validation worlds overlap"
    return inner_train, validation
