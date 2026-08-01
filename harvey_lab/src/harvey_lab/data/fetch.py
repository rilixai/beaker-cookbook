"""On-demand download of individual LAB task folders from GitHub.

The full ``harveyai/harvey-labs`` benchmark tree is ~2.7 GB across ~53k files,
so cloning it just to run a handful of tasks is wasteful. This module fetches
*only* the task directories a run actually needs (the chosen split, capped by
``--limit``) from GitHub at the pinned ``HARVEY_LABS_COMMIT`` and caches them
locally, reproducing the same on-disk layout a clone would give
(``<tasks_root>/<task_id>/task.json`` + ``<task_id>/documents/…``) so the
loader and the agent's workspace factory work unchanged.

:func:`ensure_task_dirs` is safe to call concurrently — from several threads
and from several processes sharing one cache. Each task is downloaded under a
per-task file lock into a private staging tree and only then renamed into
``tasks/<task_id>``, so a task path that exists is always a complete tree: no
other caller can ever observe a half-written directory, stray ``.part`` files,
or have the tree it is reading deleted underneath it. "Already cached" is
decided by re-checking the recorded file manifest (every path at its expected
size), not a bare sentinel, so an interrupted download is never mistaken for a
finished one.

A task's whole file tree is enumerated with a single git Trees API call
(``recursive=1``) rather than one contents-API call per subdirectory — this
matters because the unauthenticated API limit is only 60 req/hour and a large
task has hundreds of nested folders. File bodies then download in parallel
from ``raw.githubusercontent.com``, which is not API-rate-limited. Set
``GITHUB_TOKEN`` to raise the API limit to 5000/hour. Already-fetched tasks are
skipped via their manifest marker, so re-runs hit the network only for tasks
that are missing or incomplete.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import os
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any


try:  # POSIX only; the in-process lock still serializes threads without it.
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]

from ..config import HARVEY_LABS_COMMIT


logger = logging.getLogger(__name__)

REPO = "harveyai/harvey-labs"
_CONTENTS_API = "https://api.github.com/repos/{repo}/contents/{path}?ref={ref}"
_TREES_API = "https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1"
_RAW_URL = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"
_USER_AGENT = "harvey-lab-cookbook"
_MAX_RETRIES = 3
_DOWNLOAD_WORKERS = 8

# Many LAB task ids share the same parent directory (e.g.
# contracts/commercial-vendor-customer/...), so the contents listing for that
# directory can be reused across tasks rather than re-fetched.
_contents_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
_contents_lock = threading.Lock()

# One lock per (cache root, task id) so threads in this process serialize on a
# task exactly like separate processes do via the on-disk lock file.
_task_locks: dict[tuple[str, str], threading.Lock] = {}
# (cache root, task id, commit) triples whose manifest has already been verified
# on disk in this process, so repeated calls skip re-stat'ing thousands of files.
_verified: set[tuple[str, str, str]] = set()
_registry_lock = threading.Lock()


def default_cache_dir() -> Path:
    """Root of the local task cache (``$HARVEY_LAB_CACHE`` or ``~/.cache``)."""
    base = os.environ.get("HARVEY_LAB_CACHE")
    root = Path(base) if base else Path.home() / ".cache" / "harvey_lab"
    return root / HARVEY_LABS_COMMIT


def _request(url: str) -> bytes:
    headers = {"User-Agent": _USER_AGENT}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed https host
                return bytes(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 403 and "rate limit" in (exc.read().decode("utf-8", "replace").lower()):
                raise RuntimeError(
                    "GitHub API rate limit hit while fetching LAB tasks. Set GITHUB_TOKEN "
                    "(a classic PAT, no scopes needed for public repos) to raise the limit "
                    "to 5000/hour, or pass --tasks-root pointing at a local clone."
                ) from exc
            if exc.code in (429, 500, 502, 503) and attempt < _MAX_RETRIES - 1:
                last_exc = exc
                time.sleep(2**attempt)
                continue
            raise
        except urllib.error.URLError as exc:
            if attempt < _MAX_RETRIES - 1:
                last_exc = exc
                time.sleep(2**attempt)
                continue
            raise
    raise RuntimeError(f"Failed to GET {url}: {last_exc}")


def _list_contents(repo_path: str, ref: str) -> list[dict[str, Any]]:
    """Return the GitHub contents-API listing for directory ``repo_path``.

    Results are cached per ``(repo_path, ref)`` so sibling tasks fetched in the
    same run share one API call.
    """
    key = (repo_path, ref)
    with _contents_lock:
        if key in _contents_cache:
            return _contents_cache[key]

    # Percent-encode the path (LAB folders can contain spaces, e.g.
    # "1.0 Transaction Documents"); keep "/" as the path separator.
    quoted = urllib.parse.quote(repo_path, safe="/")
    listing = json.loads(_request(_CONTENTS_API.format(repo=REPO, path=quoted, ref=ref)))
    if not isinstance(listing, list):
        raise RuntimeError(f"Expected a directory listing at {repo_path!r}, got a file.")

    with _contents_lock:
        _contents_cache[key] = listing
    return listing


def _subtree_sha(task_id: str, ref: str) -> str:
    """Tree SHA of ``tasks/<task_id>`` (one contents call on its parent dir)."""
    parent, _, name = task_id.rpartition("/")
    parent_path = f"tasks/{parent}" if parent else "tasks"
    for entry in _list_contents(parent_path, ref):
        if entry["name"] == name and entry["type"] == "dir":
            return str(entry["sha"])
    raise FileNotFoundError(f"Task {task_id!r} not found under tasks/ at {ref[:10]}.")


def _walk_blob_paths(repo_path: str, ref: str, prefix: str = "") -> list[tuple[str, int]]:
    """Enumerate ``(path, size)`` (path relative to ``repo_path``) via contents."""
    blobs: list[tuple[str, int]] = []
    for entry in _list_contents(repo_path, ref):
        rel = f"{prefix}{entry['name']}"
        if entry["type"] == "dir":
            blobs.extend(_walk_blob_paths(f"{repo_path}/{entry['name']}", ref, rel + "/"))
        elif entry["type"] == "file":
            blobs.append((rel, int(entry.get("size", 0))))
    return blobs


def _blob_paths(task_id: str, ref: str) -> list[tuple[str, int]]:
    """``(path, size)`` for files under ``tasks/<task_id>``, via the Trees API.

    One recursive Trees call lists the whole subtree regardless of nesting;
    falls back to a per-directory walk only if GitHub truncates the response
    (subtrees over ~100k entries — no LAB task is that large). ``size`` lets the
    downloader skip files already fully on disk so an interrupted big task
    (the ~11 diligence data-rooms have 3k–4k files) resumes instead of restarts.
    """
    tree = json.loads(_request(_TREES_API.format(repo=REPO, sha=_subtree_sha(task_id, ref))))
    if tree.get("truncated"):
        return _walk_blob_paths(f"tasks/{task_id}", ref)
    return [(item["path"], int(item.get("size", 0))) for item in tree.get("tree", []) if item["type"] == "blob"]


def _download_task(
    task_id: str,
    dest: Path,
    ref: str,
    on_file: Callable[[], None] | None = None,
) -> list[tuple[str, int]]:
    """Download every file under ``tasks/<task_id>`` into ``dest`` in parallel.

    ``dest`` is a private staging directory, never a path another caller reads.
    Resumable: a file already on disk at its expected byte size is skipped, and
    each download lands via a temp file + atomic rename, so a file with the
    final name is always complete. ``on_file`` is called once per file (skipped
    or downloaded) for progress reporting. Returns the ``(path, size)`` manifest
    that was fetched.
    """
    blobs = _blob_paths(task_id, ref)
    dest.mkdir(parents=True, exist_ok=True)

    def _fetch_one(rel: str, size: int) -> None:
        target = dest / rel
        if not (target.is_file() and target.stat().st_size == size):
            repo_path = f"tasks/{task_id}/{rel}"
            url = _RAW_URL.format(repo=REPO, ref=ref, path=urllib.parse.quote(repo_path, safe="/"))
            data = _request(url)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            tmp.write_bytes(data)
            tmp.replace(target)
        if on_file is not None:
            on_file()

    with concurrent.futures.ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as pool:
        for fut in concurrent.futures.as_completed([pool.submit(_fetch_one, rel, size) for rel, size in blobs]):
            fut.result()  # surface the first download error
    return blobs


def _task_relpath(task_id: str) -> Path:
    """Validate ``task_id`` as a relative cache subpath (ids may be nested)."""
    normalized = task_id.strip().strip("/")
    parts = [part for part in normalized.split("/") if part]
    if not parts or any(part in (".", "..") for part in parts) or Path(normalized).is_absolute():
        raise ValueError(f"Invalid LAB task id: {task_id!r}")
    return Path(*parts)


@contextmanager
def _task_lock(root: Path, task_id: str) -> Iterator[None]:
    """Serialize work on one task across threads *and* across processes.

    The lock file is named by a digest of the task id so a nested id needs no
    directory tree of its own, and it lives outside ``tasks/`` so it can never
    be mistaken for task content.
    """
    key = (str(root), task_id)
    with _registry_lock:
        lock = _task_locks.setdefault(key, threading.Lock())
    with lock:
        if fcntl is None:  # pragma: no cover - non-POSIX
            yield
            return
        locks_dir = root / ".locks"
        locks_dir.mkdir(parents=True, exist_ok=True)
        lock_path = locks_dir / f"{hashlib.sha256(task_id.encode()).hexdigest()[:32]}.lock"
        with lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_marker(marker: Path) -> dict[str, Any] | None:
    """Parse a completion marker, or ``None`` if absent/legacy/corrupt."""
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _manifest_of(marker_payload: Mapping[str, Any]) -> list[tuple[str, int]]:
    files = marker_payload.get("files")
    if not isinstance(files, list):
        return []
    return [(str(entry[0]), int(entry[1])) for entry in files if isinstance(entry, (list, tuple)) and len(entry) == 2]


def _tree_matches_manifest(tree: Path, manifest: Sequence[tuple[str, int]]) -> bool:
    """True only if every manifest entry is on disk at its recorded size."""
    if not manifest:
        return False
    for rel, size in manifest:
        path = tree / rel
        try:
            if not path.is_file() or path.stat().st_size != size:
                return False
        except OSError:
            return False
    return True


def _is_cached(root: Path, task_id: str, commit: str) -> bool:
    """Whether ``tasks/<task_id>`` is a complete tree fetched at ``commit``.

    Deliberately stricter than a sentinel check: a marker written by an older
    version, one recorded at a different commit, or a tree missing any manifest
    file (a download interrupted mid-flight, a hand-deleted document) all count
    as *not* cached, so the task is re-materialized rather than handed to the
    agent half-populated.
    """
    key = (str(root), task_id, commit)
    with _registry_lock:
        if key in _verified:
            return True
    relpath = _task_relpath(task_id)
    payload = _read_marker(root / ".fetched" / relpath)
    if payload is None or payload.get("commit") != commit:
        return False
    tree = root / "tasks" / relpath
    if not (tree / "task.json").is_file() or not _tree_matches_manifest(tree, _manifest_of(payload)):
        return False
    with _registry_lock:
        _verified.add(key)
    return True


def _discard(path: Path) -> None:
    """Remove ``path`` (a private staging/superseded tree) if it exists."""
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        with suppress(OSError):
            path.unlink()


def _prune_empty_parents(leaf: Path, stop: Path) -> None:
    """Remove now-empty staging parents (nested ids leave a dir per level)."""
    current = leaf
    while current != stop and stop in current.parents:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _install_tree(stage: Path, dest: Path, trash_root: Path) -> None:
    """Move a completed staging tree to ``dest`` atomically.

    Any tree already at ``dest`` is *moved* aside and deleted afterwards rather
    than deleted in place, so a reader that is walking it keeps a coherent tree
    and ``dest`` is never a partially-emptied directory.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    trash_root.mkdir(parents=True, exist_ok=True)
    superseded = trash_root / f"{dest.name}-{uuid.uuid4().hex[:8]}"
    if dest.exists():
        dest.rename(superseded)
    try:
        stage.rename(dest)
    except OSError:
        if superseded.exists():  # put the old tree back rather than leaving a hole
            superseded.rename(dest)
        raise
    _discard(superseded)


def _materialize_task(root: Path, task_id: str, commit: str, label: str) -> None:
    """Fetch one task into the cache, if it is not already complete there.

    Runs entirely under the task lock: the cached check is re-evaluated after
    acquiring it, so of N concurrent callers exactly one downloads and the rest
    return as soon as it lands.
    """
    relpath = _task_relpath(task_id)
    dest = root / "tasks" / relpath
    stage_root = root / ".partial" / relpath
    stage_tree = stage_root / "tree"
    stage_commit = stage_root / "commit"

    with _task_lock(root, task_id):
        if _is_cached(root, task_id, commit):
            return
        # Staging is private to the lock holder, so anything left there by an
        # interrupted run is either resumable (same commit) or junk.
        if stage_root.exists() and (
            not stage_commit.is_file() or stage_commit.read_text(encoding="utf-8").strip() != commit
        ):
            _discard(stage_root)
        stage_root.mkdir(parents=True, exist_ok=True)
        stage_commit.write_text(commit, encoding="utf-8")

        logger.info("  %s %s ...", label, task_id)
        count = 0
        counter_lock = threading.Lock()

        def _tick() -> None:
            # Log a running file count so a big task (some have thousands of
            # documents) visibly ticks along rather than looking hung.
            nonlocal count
            with counter_lock:
                count += 1
                done = count
            if done % 25 == 0:
                logger.info("      ... %d files", done)

        manifest = _download_task(task_id, stage_tree, commit, _tick)
        if not (stage_tree / "task.json").is_file():
            raise FileNotFoundError(f"Fetched {task_id} but no task.json — is the task id valid at {commit[:10]}?")
        if not _tree_matches_manifest(stage_tree, manifest):
            raise RuntimeError(f"Fetched {task_id} but the tree is incomplete against the source manifest.")
        for leftover in stage_tree.rglob("*.part"):
            leftover.unlink(missing_ok=True)

        _install_tree(stage_tree, dest, root / ".trash")
        # The marker is written only once the complete tree is in place, and it
        # carries the manifest so completeness is re-checkable later.
        marker = root / ".fetched" / relpath
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps({"commit": commit, "files": [[rel, size] for rel, size in manifest]}),
            encoding="utf-8",
        )
        _discard(stage_root)
        _prune_empty_parents(stage_root.parent, root / ".partial")
        with _registry_lock:
            _verified.add((str(root), task_id, commit))
        logger.info("  %s %s done (%d files)", label, task_id, count)


def ensure_task_dirs(
    task_ids: Sequence[str],
    *,
    commit: str = HARVEY_LABS_COMMIT,
    cache_dir: Path | None = None,
) -> Path:
    """Download each ``tasks/<task_id>`` folder into the cache; return the tasks root.

    Safe to call concurrently for overlapping task ids: each task is fetched
    under a per-task lock (in-process and cross-process) into a private staging
    tree that is renamed into ``tasks/<task_id>`` only once complete, so a task
    path is either absent or a whole tree — never mid-download, never deleted
    under a reader, never littered with ``.part`` files. Individual file
    downloads stay resumable, and an interrupted task resumes from its staging
    tree on the next call rather than restarting.

    Only missing or incomplete tasks are fetched: a task counts as cached only
    when its ``.fetched/<task_id>`` marker records this ``commit`` *and* every
    file in that marker's manifest is on disk at its recorded size. The
    returned path is a ``tasks/`` directory usable as ``--tasks-root`` —
    ``<tasks_root>/<task_id>/{task.json,documents/…}``.
    """
    root = cache_dir or default_cache_dir()
    tasks_root = root / "tasks"

    unique = list(dict.fromkeys(task_ids))
    pending = [tid for tid in unique if not _is_cached(root, tid, commit)]
    if pending:
        logger.info("Fetching %d task(s) from %s@%s into %s ...", len(pending), REPO, commit[:10], root)
    for index, task_id in enumerate(pending, start=1):
        _materialize_task(root, task_id, commit, f"[{index}/{len(pending)}]")
    tasks_root.mkdir(parents=True, exist_ok=True)
    return tasks_root
