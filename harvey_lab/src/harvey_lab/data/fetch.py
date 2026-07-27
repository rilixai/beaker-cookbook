"""On-demand download of individual LAB task folders from GitHub.

The full ``harveyai/harvey-labs`` benchmark tree is ~2.7 GB across ~53k files,
so cloning it just to run a handful of tasks is wasteful. This module fetches
*only* the task directories a run actually needs (the chosen split, capped by
``--limit``) from GitHub at the pinned ``HARVEY_LABS_COMMIT`` and caches them
locally, reproducing the same on-disk layout a clone would give
(``<tasks_root>/<task_id>/task.json`` + ``<task_id>/documents/…``) so the
loader and the agent's workspace factory work unchanged.

A task's whole file tree is enumerated with a single git Trees API call
(``recursive=1``) rather than one contents-API call per subdirectory — this
matters because the unauthenticated API limit is only 60 req/hour and a large
task has hundreds of nested folders. File bodies then download in parallel
from ``raw.githubusercontent.com``, which is not API-rate-limited. Set
``GITHUB_TOKEN`` to raise the API limit to 5000/hour. Already-fetched tasks are
skipped via a sentinel marker, so re-runs hit the network only for new tasks.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..config import HARVEY_LABS_COMMIT


logger = logging.getLogger(__name__)

REPO = "harveyai/harvey-labs"
_CONTENTS_API = "https://api.github.com/repos/{repo}/contents/{path}?ref={ref}"
_TREES_API = "https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1"
_RAW_URL = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"
_USER_AGENT = "harvey-lab-cookbook"
_MAX_RETRIES = 3
_DOWNLOAD_WORKERS = 8


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
    """Return the GitHub contents-API listing for directory ``repo_path``."""
    # Percent-encode the path (LAB folders can contain spaces, e.g.
    # "1.0 Transaction Documents"); keep "/" as the path separator.
    quoted = urllib.parse.quote(repo_path, safe="/")
    listing = json.loads(_request(_CONTENTS_API.format(repo=REPO, path=quoted, ref=ref)))
    if not isinstance(listing, list):
        raise RuntimeError(f"Expected a directory listing at {repo_path!r}, got a file.")
    return listing


def _subtree_sha(task_id: str, ref: str) -> str:
    """Tree SHA of ``tasks/<task_id>`` (one contents call on its parent dir)."""
    parent, _, name = task_id.rpartition("/")
    parent_path = f"tasks/{parent}" if parent else "tasks"
    for entry in _list_contents(parent_path, ref):
        if entry["name"] == name and entry["type"] == "dir":
            return str(entry["sha"])
    raise FileNotFoundError(f"Task {task_id!r} not found under tasks/ at {ref[:10]}.")


def _walk_blob_paths(repo_path: str, ref: str, prefix: str = "") -> list[str]:
    """Enumerate file paths (relative to ``repo_path``) via the contents API."""
    paths: list[str] = []
    for entry in _list_contents(repo_path, ref):
        rel = f"{prefix}{entry['name']}"
        if entry["type"] == "dir":
            paths.extend(_walk_blob_paths(f"{repo_path}/{entry['name']}", ref, rel + "/"))
        elif entry["type"] == "file":
            paths.append(rel)
    return paths


def _blob_paths(task_id: str, ref: str) -> list[str]:
    """File paths under ``tasks/<task_id>`` (relative to it), via the Trees API.

    One recursive Trees call lists the whole subtree regardless of nesting;
    falls back to a per-directory walk only if GitHub truncates the response
    (subtrees over ~100k entries — no LAB task is that large).
    """
    tree = json.loads(_request(_TREES_API.format(repo=REPO, sha=_subtree_sha(task_id, ref))))
    if tree.get("truncated"):
        return _walk_blob_paths(f"tasks/{task_id}", ref)
    return [item["path"] for item in tree.get("tree", []) if item["type"] == "blob"]


def _download_task(task_id: str, dest: Path, ref: str, on_file: Callable[[], None] | None = None) -> None:
    """Download every file under ``tasks/<task_id>`` into ``dest`` in parallel.

    ``on_file`` is called once per downloaded file (for progress reporting).
    """
    rel_paths = _blob_paths(task_id, ref)
    dest.mkdir(parents=True, exist_ok=True)

    def _fetch_one(rel: str) -> None:
        repo_path = f"tasks/{task_id}/{rel}"
        url = _RAW_URL.format(repo=REPO, ref=ref, path=urllib.parse.quote(repo_path, safe="/"))
        data = _request(url)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        if on_file is not None:
            on_file()

    with concurrent.futures.ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as pool:
        for fut in concurrent.futures.as_completed([pool.submit(_fetch_one, rel) for rel in rel_paths]):
            fut.result()  # surface the first download error


def ensure_task_dirs(
    task_ids: Sequence[str],
    *,
    commit: str = HARVEY_LABS_COMMIT,
    cache_dir: Path | None = None,
) -> Path:
    """Download each ``tasks/<task_id>`` folder into the cache; return the tasks root.

    Only missing tasks are fetched (a ``.fetched/<task_id>`` sentinel marks a
    completed download). The returned path is a ``tasks/`` directory usable as
    ``--tasks-root`` — ``<tasks_root>/<task_id>/{task.json,documents/…}``.
    """
    root = cache_dir or default_cache_dir()
    tasks_root = root / "tasks"
    sentinels = root / ".fetched"

    def _cached_for(tid: str) -> bool:
        # The sentinel records the commit it was fetched at; a mismatch (e.g. a
        # reused --cache-dir after a pin bump) must refetch, not serve stale
        # trees. Also require the task tree to still be on disk (guards a
        # manually-deleted cache dir leaving a dangling sentinel).
        marker = sentinels / tid
        if not (marker.is_file() and marker.read_text(encoding="utf-8") == commit):
            return False
        return (tasks_root / tid / "task.json").is_file()

    pending = [tid for tid in task_ids if not _cached_for(tid)]
    if pending:
        logger.info("Fetching %d task(s) from %s@%s into %s ...", len(pending), REPO, commit[:10], root)
    for i, tid in enumerate(pending, start=1):
        dest = tasks_root / tid
        # Drop any stale tree (different commit) so removed files don't linger.
        if dest.exists():
            shutil.rmtree(dest)
        logger.info("  [%d/%d] %s ...", i, len(pending), tid)

        # Log a running file count so a big task (some have thousands of
        # documents) visibly ticks along rather than looking hung. Downloads
        # run on a thread pool, so guard the counter with a lock.
        count = 0
        lock = threading.Lock()

        def _tick() -> None:
            nonlocal count
            with lock:
                count += 1
                done = count
            if done % 25 == 0:
                logger.info("      ... %d files", done)

        _download_task(tid, dest, commit, _tick)
        logger.info("  [%d/%d] %s done (%d files)", i, len(pending), tid, count)
        if not (dest / "task.json").is_file():
            raise FileNotFoundError(f"Fetched {tid} but no task.json — is the task id valid at {commit[:10]}?")
        marker = sentinels / tid
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(commit, encoding="utf-8")
    return tasks_root
