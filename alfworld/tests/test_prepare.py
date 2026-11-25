"""Preparation: checksums, upstream filtering, disjoint deterministic splits, counts, task types.

Synthetic archives stand in for the pinned release, so these tests run the
unmodified upstream ``collect_game_files`` without the 140 MB download.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import pytest
from alfworld.agents.environment.alfred_tw_env import TASK_TYPES as UPSTREAM_TASK_TYPES

from alfworld_agent import data
from alfworld_agent.data import ARCHIVE_ROOT, SPLIT_FILES, TASK_TYPES, Source, load_rows, prepare, task_type_of


def folders(split: str, per_type: int) -> list[str]:
    return [f"{split}/{t}-Obj{i}-None-Recep-{i}/trial_T2019_{i:06d}" for t in TASK_TYPES for i in range(per_type)]


def write_zip(path: Path, files: dict[str, str]) -> Source:
    with zipfile.ZipFile(path, "w") as zipped:
        for name, text in files.items():
            zipped.writestr(f"{ARCHIVE_ROOT}/{name}", text)
    return Source(path.as_uri(), hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """A tiny release: 4 train, 2 valid_seen and 1 valid_unseen game per task type, plus games upstream drops."""
    kept = folders("train", 4) + folders("valid_seen", 2) + folders("valid_unseen", 1)
    games = {f"{f}/game.tw-pddl": json.dumps({"solvable": True}) for f in kept}
    trajs = {f"{f}/traj_data.json": json.dumps({"task_type": f.split("/")[1].split("-")[0]}) for f in kept}
    dropped = {
        "train/pick_and_place_simple-AppleSliced-None-Fridge-1/trial_T1": True,  # sliced: upstream drops it
        "train/pick_and_place_simple-Unsolvable-None-Fridge-2/trial_T2": False,  # marked unsolvable
    }
    for folder, solvable in dropped.items():
        games[f"{folder}/game.tw-pddl"] = json.dumps({"solvable": solvable})
        trajs[f"{folder}/traj_data.json"] = json.dumps({"task_type": "pick_and_place_simple"})
    trajs["train/pick_and_place_simple-NoGame-None-Fridge-3/trial_T3/traj_data.json"] = "{}"  # no game file
    sources = {
        "games": write_zip(tmp_path / "games.zip", games),
        "json": write_zip(tmp_path / "json.zip", trajs),
        "pddl": write_zip(tmp_path / "pddl.zip", {"README": "unused"}),
    }
    monkeypatch.setattr(data, "SOURCES", sources)
    expected = {"train": 4 * 6, "valid_seen": 2 * 6, "valid_unseen": 6}
    monkeypatch.setattr(data, "EXPECTED_COUNTS", expected)
    return expected


def test_task_types_match_upstream() -> None:
    assert set(TASK_TYPES) == set(UPSTREAM_TASK_TYPES.values())


def test_a_checksum_mismatch_fails(tmp_path: Path, release: dict[str, int]) -> None:
    cached = tmp_path / "cache" / "downloads" / data.SOURCES["games"].filename
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        prepare(tmp_path / "out", tmp_path / "cache")
    assert not (tmp_path / "out").exists()


def test_splits_are_upstream_filtered_disjoint_deterministic_and_match_the_manifest(
    tmp_path: Path, release: dict[str, int]
) -> None:
    prepare(tmp_path / "a", tmp_path / "cache")
    prepare(tmp_path / "b", tmp_path / "cache")
    for file in [*SPLIT_FILES.values(), "manifest.json"]:
        assert (tmp_path / "a" / file).read_bytes() == (tmp_path / "b" / file).read_bytes()

    manifest = json.loads((tmp_path / "a" / "manifest.json").read_text())
    rows = {name: load_rows(tmp_path / "a" / file) for name, file in SPLIT_FILES.items()}
    assert (
        manifest["counts"] == {name: len(r) for name, r in rows.items()} == {"train": 24, "validation": 12, "test": 6}
    )
    for name, members in rows.items():
        assert manifest["counts_by_task_type"][name] == dict(Counter(r.expected.task_type for r in members))
        assert {r.input.split for r in members} == {manifest["upstream_splits"][name]}
        for r in members:
            assert task_type_of(r.input.game) == r.expected.task_type
    games = [r.input.game for members in rows.values() for r in members]
    assert len(games) == len(set(games))
    assert not any("Sliced" in g or "Unsolvable" in g or "NoGame" in g for g in games)
    # Any prefix is a fair sample: the first six training games cover all six task types.
    assert {r.expected.task_type for r in rows["train"][:6]} == set(TASK_TYPES)


def test_prepare_fails_when_upstream_counts_change(
    tmp_path: Path, release: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data, "EXPECTED_COUNTS", {**release, "valid_unseen": 134})
    with pytest.raises(ValueError, match="counts changed"):
        prepare(tmp_path / "out", tmp_path / "cache")


def test_games_per_type_keeps_the_first_games_of_each_type(tmp_path: Path, release: dict[str, int]) -> None:
    full = prepare(tmp_path / "full", tmp_path / "cache")
    small = prepare(tmp_path / "small", tmp_path / "cache", games_per_type=2)
    assert small["counts"] == {"train": 12, "validation": 12, "test": 6}
    assert set(small["counts_by_task_type"]["train"].values()) == {2}
    for file in SPLIT_FILES.values():
        kept = (tmp_path / "small" / file).read_text().splitlines()
        assert kept == [line for line in (tmp_path / "full" / file).read_text().splitlines() if line in kept]
    assert json.loads((tmp_path / "small" / "manifest.json").read_text())["games_per_type"] == 2
    assert full["counts"]["train"] == 24


def test_task_type_parses_from_the_folder_name() -> None:
    game = "valid_unseen/pick_two_obj_and_place-Pencil-None-Desk-1/trial_T20190909_1/game.tw-pddl"
    assert task_type_of(game) == "pick_two_obj_and_place"
    with pytest.raises(ValueError):
        task_type_of("valid_unseen/unknown_type-Pencil-None-Desk-1/trial_T1/game.tw-pddl")
