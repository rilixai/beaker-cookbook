"""Fetch the checksum-pinned ALFWorld game files and write reproducible splits.

One case is one game (one full episode). The splits are ALFWorld's own, after
ALFWorld's own filtering (``AlfredTWEnv.collect_game_files``):

- ``optimization/train.jsonl``: upstream ``train``
- ``optimization/test.jsonl``: upstream ``valid_seen``, the validation set
- ``heldout/test.jsonl``: upstream ``valid_unseen``, for final evaluation only

Each file is ordered by a seeded hash, stratified by task type, so any prefix
is a fair sample. Game files are never committed: ``prepare`` and the Beaker
sandbox download and verify them.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal
from urllib.request import urlopen

from pydantic import BaseModel, ConfigDict, Field, model_validator


ALFWORLD_VERSION = "0.4.2"


@dataclass(frozen=True)
class Source:
    url: str
    sha256: str

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


RELEASES = "https://github.com/alfworld/alfworld/releases/download"
SOURCES = {
    # ALFRED trajectory metadata: prepare reads each game's traj_data.json, as upstream filtering does.
    "json": Source(
        f"{RELEASES}/0.2.2/json_2.1.1_json.zip", "25171f16e20ad7b048c47275c45b0babf3aa1cbab29cec97387922350a9844bc"
    ),
    # ALFRED initial-state PDDL, part of the upstream data download. The text env does not read it.
    "pddl": Source(
        f"{RELEASES}/0.2.2/json_2.1.1_pddl.zip", "913942ebed06659ea0da2f8122512d98bc6add30d84961ca803132d8fbcad585"
    ),
    # The playable TextWorld games, one game.tw-pddl per trajectory. The only file the sandbox needs.
    "games": Source(
        f"{RELEASES}/0.4.2/json_2.1.3_tw-pddl.zip", "5df77ea759f2211a4106082839ddbbb790f1ba4e7d097ed732cf453f72aa36cf"
    ),
}
ARCHIVE_ROOT = "json_2.1.1"
CACHE = Path(os.environ.get("ALFWORLD_AGENT_CACHE", Path.home() / ".cache" / "alfworld-agent"))

TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)
TaskType = Literal[
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
]
UpstreamSplit = Literal["train", "valid_seen", "valid_unseen"]

SPLIT_SEED = "alfworld-v1"
# Recipe split -> (upstream split, file, upstream train_eval mode for collect_game_files).
SPLITS: dict[str, tuple[UpstreamSplit, str, str]] = {
    "train": ("train", "optimization/train.jsonl", "train"),
    "validation": ("valid_seen", "optimization/test.jsonl", "eval_in_distribution"),
    "test": ("valid_unseen", "heldout/test.jsonl", "eval_out_of_distribution"),
}
SPLIT_FILES = {name: file for name, (_, file, _) in SPLITS.items()}
# Game counts after upstream filtering, measured on the pinned files. prepare fails if they change.
EXPECTED_COUNTS = {"train": 3553, "valid_seen": 140, "valid_unseen": 134}


def task_type_of(game: str) -> TaskType:
    """The task type from the game's folder name, e.g. ``valid_seen/pick_and_place_simple-Book-.../trial_.../``."""
    parts = game.split("/")
    if len(parts) != 4 or parts[-1] != "game.tw-pddl" or not parts[2].startswith("trial_"):
        raise ValueError(f"Not a game path: {game!r}")
    prefix = parts[1].split("-", 1)[0]
    if prefix not in TASK_TYPES:
        raise ValueError(f"Unknown task type {prefix!r} in {game!r}")
    return prefix  # type: ignore[return-value]


def row_id(split: str, game: str) -> str:
    return f"alfworld-{split}-{hashlib.sha256(game.encode()).hexdigest()[:12]}"


class GameRef(BaseModel):
    """Which game to play. Never shown to the agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    split: UpstreamSplit
    game: str = Field(pattern=r"^(train|valid_seen|valid_unseen)/[^/]+/trial_[^/]+/game\.tw-pddl$")

    @model_validator(mode="after")
    def game_in_split(self) -> GameRef:
        if not self.game.startswith(f"{self.split}/"):
            raise ValueError("The game path must be inside its split")
        task_type_of(self.game)
        return self


class Expected(BaseModel):
    """Used only for reporting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_type: TaskType


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^alfworld-(train|valid_seen|valid_unseen)-[0-9a-f]{12}$")
    input: GameRef
    expected: Expected

    @model_validator(mode="after")
    def consistent(self) -> Row:
        if self.id != row_id(self.input.split, self.input.game):
            raise ValueError("The row ID must be the hash of its game path")
        if self.expected.task_type != task_type_of(self.input.game):
            raise ValueError("The task type must match the game's folder name")
        return self


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name: str, cache: Path = CACHE) -> Path:
    """One pinned release file, from cache when present. Any checksum mismatch fails closed."""
    source = SOURCES[name]
    path = cache / "downloads" / source.filename
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        with urlopen(source.url, timeout=300) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        partial.replace(path)
    if _sha256(path) != source.sha256:
        raise ValueError(f"Checksum mismatch for {source.filename}; delete {path} and retry. No data was prepared.")
    return path


@contextlib.contextmanager
def _locked(cache: Path) -> Iterator[None]:
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "games.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def games_root(cache: Path = CACHE) -> Path:
    """The extracted game files (``<root>/<split>/<task folder>/<trial>/game.tw-pddl``), fetched once.

    Safe to call from several processes at once; later calls return immediately.
    """
    target = cache / f"games-{SOURCES['games'].sha256[:12]}"
    with _locked(cache):
        if not (target / ".complete").exists():
            archive = fetch("games", cache)
            staging = Path(tempfile.mkdtemp(dir=cache, prefix="games-staging-"))
            with zipfile.ZipFile(archive) as zipped:
                if any(not name.startswith(f"{ARCHIVE_ROOT}/") for name in zipped.namelist()):
                    raise ValueError(f"Unexpected layout in {archive.name}")
                zipped.extractall(staging)
            (staging / ".complete").touch()
            shutil.rmtree(target, ignore_errors=True)
            staging.rename(target)
    return target / ARCHIVE_ROOT


def collect_games(cache: Path = CACHE) -> dict[UpstreamSplit, list[str]]:
    """Game paths per split, relative to ``games_root``, as upstream ``collect_game_files`` selects them.

    Upstream walks folders holding both ``traj_data.json`` (from the json
    archive) and ``game.tw-pddl``. This builds that tree in a temporary
    folder, with only the traj_data.json files of folders that have a game,
    which upstream would skip anyway, and runs the unmodified upstream filter.
    """
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    root = games_root(cache)
    json_archive = fetch("json", cache)
    splits = [upstream for upstream, _, _ in SPLITS.values()]
    game_dirs = {path.parent.relative_to(root).as_posix() for s in splits for path in (root / s).rglob("game.tw-pddl")}
    with tempfile.TemporaryDirectory(dir=cache, prefix="collect-") as temporary:
        tree = Path(temporary) / ARCHIVE_ROOT
        with zipfile.ZipFile(json_archive) as zipped:
            for name in zipped.namelist():
                folder = name.removeprefix(f"{ARCHIVE_ROOT}/").rpartition("/")[0]
                if name.endswith("/traj_data.json") and folder in game_dirs:
                    zipped.extract(name, temporary)
        for folder in game_dirs:
            (tree / folder).mkdir(parents=True, exist_ok=True)
            (tree / folder / "game.tw-pddl").symlink_to(root / folder / "game.tw-pddl")
        config = {
            "env": {"goal_desc_human_anns_prob": 0, "task_types": [1, 2, 3, 4, 5, 6]},
            "dataset": {
                "data_path": str(tree / "train"),
                "eval_id_data_path": str(tree / "valid_seen"),
                "eval_ood_data_path": str(tree / "valid_unseen"),
                "num_train_games": -1,
                "num_eval_games": -1,
            },
        }
        selected: dict[UpstreamSplit, list[str]] = {}
        for upstream, _, mode in SPLITS.values():
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                files = AlfredTWEnv(config, mode).game_files
            selected[upstream] = sorted(Path(f).relative_to(tree).as_posix() for f in files)
    return selected


def _hash(purpose: str, key: str) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}:{purpose}:{key}".encode()).hexdigest()


def fair_order(rows: Sequence[Row]) -> list[Row]:
    """Seeded-hash order, stratified by task type, so any prefix is a fair sample.

    Within a task type, games are ordered by a seeded hash; the i-th of n sits
    at position (i + 0.5) / n. Sorting every game by that position interleaves
    the six types in proportion.
    """
    by_type: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_type[row.expected.task_type].append(row)
    placed: list[tuple[float, str, Row]] = []
    for members in by_type.values():
        members.sort(key=lambda r: _hash("order", r.id))
        placed.extend(((i + 0.5) / len(members), _hash("tie", r.id), r) for i, r in enumerate(members))
    return [row for _, _, row in sorted(placed, key=lambda item: item[:2])]


def make_rows(split: UpstreamSplit, games: Sequence[str]) -> list[Row]:
    return [
        Row(
            id=row_id(split, game),
            input=GameRef(split=split, game=game),
            expected=Expected(task_type=task_type_of(game)),
        )
        for game in games
    ]


def first_per_type(rows: Sequence[Row], n: int) -> list[Row]:
    """The first ``n`` games of each task type, in file order (a seeded hash, so a fair sample)."""
    taken: Counter[str] = Counter()
    kept = []
    for row in rows:
        if taken[row.expected.task_type] < n:
            taken[row.expected.task_type] += 1
            kept.append(row)
    return kept


def prepare(output_dir: Path, cache: Path = CACHE, *, games_per_type: int | None = None) -> dict[str, Any]:
    for name in SOURCES:
        fetch(name, cache)
    games = collect_games(cache)
    counts = {split: len(paths) for split, paths in games.items()}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Upstream game counts changed: {counts}, expected {EXPECTED_COUNTS}. No data was prepared.")
    splits = {name: fair_order(make_rows(upstream, games[upstream])) for name, (upstream, _, _) in SPLITS.items()}
    if games_per_type is not None:  # a small dataset for cheap trial runs
        splits = {name: first_per_type(rows, games_per_type) for name, rows in splits.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = output_dir / SPLIT_FILES[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(row.model_dump_json() + "\n" for row in rows))
    manifest: dict[str, Any] = {
        "alfworld_version": ALFWORLD_VERSION,
        "textworld_version": version("textworld"),
        "sources": {name: {"url": s.url, "sha256": s.sha256} for name, s in SOURCES.items()},
        "split_seed": SPLIT_SEED,
        "ordered_by": "seeded hash, stratified by task_type",
        "games_per_type": games_per_type,
        "files": SPLIT_FILES,
        "upstream_splits": {name: upstream for name, (upstream, _, _) in SPLITS.items()},
        "counts": {name: len(rows) for name, rows in splits.items()},
        "counts_by_task_type": {
            name: dict(sorted(Counter(r.expected.task_type for r in rows).items())) for name, rows in splits.items()
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"counts": manifest["counts"], "counts_by_task_type": manifest["counts_by_task_type"]}


def load_rows(path: Path) -> list[Row]:
    rows = [Row.model_validate_json(line) for line in path.read_text().split("\n") if line.strip()]
    if not rows or len({row.id for row in rows}) != len(rows):
        raise ValueError("Dataset must be non-empty with unique row IDs")
    return rows


def find_row(data_dir: Path, case_id: str) -> Row:
    for file in SPLIT_FILES.values():
        for row in load_rows(data_dir / file):
            if row.id == case_id:
                return row
    raise KeyError(f"No case {case_id!r} in {data_dir}")
