"""On-demand download of individual LAB task folders from GitHub.

The full ``harveyai/harvey-labs`` benchmark tree is ~2.7 GB across ~53k files,
so cloning it just to run a handful of tasks is wasteful. This module fetches
*only* the task directories a run actually needs (the chosen split, capped by
``--limit``) from GitHub at the pinned ``HARVEY_LABS_COMMIT`` and caches them
locally, reproducing the same on-disk layout a clone would give
(``<tasks_root>/<task_id>/task.json`` + ``<task_id>/documents/…``) so the
loader and the agent's workspace factory work unchanged.

Directory listings use the GitHub contents API (rate-limited to 60 req/hour
unauthenticated — set ``GITHUB_TOKEN`` for 5000/hour when pulling large
splits); file bodies download from ``raw.githubusercontent.com``, which is not
API-rate-limited. Already-fetched tasks are skipped via a sentinel marker, so
re-runs hit the network only for new tasks.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

from ..config import HARVEY_LABS_COMMIT


logger = logging.getLogger(__name__)

REPO = "harveyai/harvey-labs"
_CONTENTS_API = "https://api.github.com/repos/{repo}/contents/{path}?ref={ref}"
_USER_AGENT = "harvey-lab-cookbook"
_MAX_RETRIES = 3


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


def _download_dir(repo_path: str, dest: Path, ref: str, on_file: Callable[[], None] | None = None) -> None:
    """Recursively download the GitHub directory ``repo_path`` into ``dest``.

    ``on_file`` is called once per downloaded file (for progress reporting).
    """
    # Percent-encode the path (LAB folders can contain spaces, e.g.
    # "1.0 Transaction Documents"); keep "/" as the path separator.
    quoted = urllib.parse.quote(repo_path, safe="/")
    listing = json.loads(_request(_CONTENTS_API.format(repo=REPO, path=quoted, ref=ref)))
    if not isinstance(listing, list):
        raise RuntimeError(f"Expected a directory listing at {repo_path!r}, got a file.")
    dest.mkdir(parents=True, exist_ok=True)
    for entry in listing:
        name = entry["name"]
        if entry["type"] == "dir":
            _download_dir(f"{repo_path}/{name}", dest / name, ref, on_file)
        elif entry["type"] == "file":
            (dest / name).write_bytes(_request(entry["download_url"]))
            if on_file is not None:
                on_file()


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
        # documents) visibly ticks along rather than looking hung.
        count = 0

        def _tick() -> None:
            nonlocal count
            count += 1
            if count % 25 == 0:
                logger.info("      ... %d files", count)

        _download_dir(f"tasks/{tid}", dest, commit, _tick)
        logger.info("  [%d/%d] %s done (%d files)", i, len(pending), tid, count)
        if not (dest / "task.json").is_file():
            raise FileNotFoundError(f"Fetched {tid} but no task.json — is the task id valid at {commit[:10]}?")
        marker = sentinels / tid
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(commit, encoding="utf-8")
    return tasks_root
