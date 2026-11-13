"""Prepare the pinned Toolathlon benchmark in local and hosted runs."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from toolathlon_baseline import UPSTREAM_REVISION
from toolathlon_baseline.prepare import MIGRATION_PATCH, prepare, prepared_source


# MCP servers and local tools backed by third-party accounts (tokens in upstream
# configs/token_key_session.py). Canvas, emails, WooCommerce and k8s are local containers.
EXTERNAL_ACCOUNT_SERVERS = frozenset(
    {
        "github",
        "google-cloud",
        "google_calendar",
        "google_forms",
        "google_map",
        "google_sheet",
        "huggingface",
        "notion",
        "snowflake",
        "wandb",
        "yahoo-finance",
        "youtube",
        "youtube-transcript",
    }
)
EXTERNAL_ACCOUNT_LOCAL_TOOLS = frozenset({"web_search"})


def _ensure_bin_on_path() -> None:
    bin_dir = str(Path(sys.executable).resolve().parent)
    current_path = os.environ.get("PATH", "")
    parts = current_path.split(os.pathsep) if current_path else []
    if bin_dir not in parts:
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{current_path}".rstrip(os.pathsep)


def _cache_dir() -> Path:
    configured = os.environ.get("BEAKER_TOOLATHLON_CACHE")
    if configured:
        return Path(configured)
    # Candidate rollouts override TMPDIR per attempt; keep the prepared upstream
    # checkout in shared /tmp so prepare_run, run_case, and score_case reuse it.
    cache_name = (
        f"beaker-toolathlon-{UPSTREAM_REVISION}-{hashlib.sha256(MIGRATION_PATCH.read_bytes()).hexdigest()[:12]}"
    )
    shared_tmp = Path("/tmp")
    if shared_tmp.is_dir() and os.access(shared_tmp, os.W_OK | os.X_OK):
        return shared_tmp / cache_name
    return Path(tempfile.gettempdir()) / cache_name


def prepare_toolathlon() -> tuple[Path, Path]:
    """Return (cache_dir, upstream_source_dir) for the pinned Toolathlon revision."""
    _ensure_bin_on_path()
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "setup.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            source = prepared_source(cache)
        except RuntimeError:
            source = prepare(cache)
        for name in ("global_configs", "token_key_session"):
            target = source / "configs" / f"{name}.py"
            example = source / "configs" / f"{name}_example.py"
            if not target.exists() and example.is_file():
                target.write_bytes(example.read_bytes())
        return cache, source


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def groundtruth_hashes(source: Path, task: str) -> dict[str, str]:
    task_dir = source / "tasks/finalpool" / task
    gt_dir = task_dir / "groundtruth_workspace"
    if gt_dir.is_dir():
        hashes = {
            path.relative_to(gt_dir).as_posix(): _sha256(path)
            for path in sorted(gt_dir.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
        }
        if hashes:
            return hashes
    eval_dir = task_dir / "evaluation"
    if not eval_dir.is_dir():
        raise ValueError(f"Missing evaluation directory for {task}: {eval_dir}")
    hashes = {
        f"evaluation/{path.relative_to(eval_dir).as_posix()}": _sha256(path)
        for path in sorted(eval_dir.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    if not hashes:
        raise ValueError(f"No groundtruth or evaluation files found in {task_dir}")
    return hashes


def load_task_reference(source: Path, task: str) -> dict[str, Any]:
    """Read the official task inputs and scoring references for a platform dataset row."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", task):
        raise ValueError(f"Unsupported Toolathlon task: {task}")
    task_dir = source / "tasks/finalpool" / task
    instruction_path = task_dir / "docs/task.md"
    config_path = task_dir / "task_config.json"
    eval_path = task_dir / "evaluation/main.py"
    if not instruction_path.is_file() or not config_path.is_file() or not eval_path.is_file():
        raise ValueError(f"Task {task} is missing required files in {task_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    instruction = instruction_path.read_text(encoding="utf-8").strip()
    gt_hashes = groundtruth_hashes(source, task)
    return {
        "id": task,
        "input": {
            "task": task,
            "instruction": instruction,
            "mcp_servers": list(config.get("needed_mcp_servers", [])),
            "local_tools": list(config.get("needed_local_tools", [])),
        },
        "expected": {
            "task": task,
            "upstream_revision": UPSTREAM_REVISION,
            "evaluation_module": f"tasks.finalpool.{task}.evaluation.main",
            "groundtruth_hashes": gt_hashes,
        },
    }
