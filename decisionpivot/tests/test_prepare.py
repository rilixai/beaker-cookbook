"""Prepare: hashes, splits by contrast group, fresh draws, determinism, nothing revealing in ``input``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from decisionpivot.data import (
    FAMILIES,
    SEEDED,
    SPLIT_FILES,
    UNSEEDED,
    load_all,
    load_rows,
    make_splits,
    release_hashes,
    sha256_file,
    split_published,
)


def all_rows(prepared: Path) -> dict[str, list[Any]]:
    return {name: load_rows(prepared / path) for name, path in SPLIT_FILES.items()}


def test_published_draws_match_release_hashes(prepared: Path, upstream: Path) -> None:
    release = release_hashes(upstream)
    for family in FAMILIES:
        assert sha256_file(upstream / "generated/tasks" / family / "items.jsonl") == release[family]
    manifest = json.loads((prepared / "manifest.json").read_text())
    assert manifest["items_sha256"]["published"] == release
    assert manifest["upstream_commit"] == "fd57688f033864e64c452561ffdfcba9739d1811"


def test_no_group_or_item_crosses_splits(prepared: Path) -> None:
    splits = all_rows(prepared)
    seen_groups: set[str] = set()
    seen_items: set[str] = set()
    for rows in splits.values():
        groups = {r.id for r in rows}
        items = {i.id for r in rows for i in r.input.items}
        assert not groups & seen_groups and not items & seen_items
        seen_groups |= groups
        seen_items |= items


def test_seeded_families_split_by_draw_and_unseeded_by_published_pool(prepared: Path) -> None:
    splits = all_rows(prepared)
    for name, rows in splits.items():
        for row in rows:
            seed_tag = ":s1:" if name == "train" else ":s2:" if name == "validation" else None
            if row.input.family in SEEDED:
                if seed_tag:
                    assert seed_tag in row.id
                else:
                    assert ":s1:" not in row.id and ":s2:" not in row.id
            else:
                assert ":s" not in row.id  # the six unseeded families use published groups only
    published_items = sum(len(r.input.items) for rows in splits.values() for r in rows if ":s" not in r.id)
    assert published_items == 13 * 600
    for family in UNSEEDED:
        sizes = {name: sum(r.input.family == family for r in rows) for name, rows in splits.items()}
        total = sum(sizes.values())
        assert abs(sizes["heldout"] / total - 0.2) < 0.03 and abs(sizes["validation"] / total - 0.2) < 0.03


def test_fresh_draws_share_no_ids_with_published(upstream: Path) -> None:
    published, fresh, _ = load_all(upstream)
    published_ids = {i.id for rows in published.values() for r in rows for i in r.input.items}
    published_groups = {r.id for rows in published.values() for r in rows}
    for draws in fresh.values():
        for rows in draws.values():
            assert not {i.id for r in rows for i in r.input.items} & published_ids
            assert not {r.id for r in rows} & published_groups


def test_splits_are_deterministic(upstream: Path, prepared: Path) -> None:
    published, fresh, _ = load_all(upstream)
    again = make_splits(published, fresh)
    for name, path in SPLIT_FILES.items():
        assert [r.id for r in again[name]] == [r.id for r in load_rows(prepared / path)]
    shuffled = list(reversed(published["policy_case"]))
    assert {k: {r.id for r in v} for k, v in split_published(shuffled).items()} == {
        k: {r.id for r in v} for k, v in split_published(published["policy_case"]).items()
    }


def test_strata_are_family_and_level(prepared: Path) -> None:
    manifest = json.loads((prepared / "manifest.json").read_text())
    for name in SPLIT_FILES:
        by_level = manifest["counts"][name]["items_by_family_level"]
        assert set(by_level) == set(FAMILIES)
        for family in FAMILIES:
            assert set(by_level[family]) == {"1", "2", "3"}


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in _keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in _keys(v)}
    return set()


def test_labels_never_in_input(prepared: Path) -> None:
    for path in SPLIT_FILES.values():
        for line in (prepared / path).read_text().splitlines():
            row = json.loads(line)
            assert set(row["input"]) == {"family", "items"}
            for item in row["input"]["items"]:
                assert set(item) == {"id", "state", "questions"}
                assert set(item["questions"]["q"]) == {"type", "instructions", "criteria"}
            # gold, level, deviation, spec and meta only live outside input
            outside_state = [{k: v for k, v in i.items() if k != "state"} for i in row["input"]["items"]]
            assert not _keys(outside_state) & {"gold", "level", "deviation", "spec", "meta", "group"}
            assert set(row["expected"]) == {"gold", "level", "layer"}
