"""Prepare and evaluate the pinned Toolathlon benchmark in local and hosted runs."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from toolathlon_baseline import QUICKSTART_TASKS, UPSTREAM_REVISION
from toolathlon_baseline.prepare import prepare, prepared_source, runtime_env


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
    shared_tmp = Path("/tmp")
    if shared_tmp.is_dir() and os.access(shared_tmp, os.W_OK | os.X_OK):
        return shared_tmp / f"beaker-toolathlon-{UPSTREAM_REVISION}"
    return Path(tempfile.gettempdir()) / f"beaker-toolathlon-{UPSTREAM_REVISION}"


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


def verify_sandbox_support() -> None:
    """Check actual enforcement in a disposable process before expensive setup."""
    probe = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("toolathlon_sandbox.py")), "--check"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if probe.returncode:
        raise RuntimeError(
            "Toolathlon onboarding requires Linux with Landlock ABI >= 3 enabled. "
            f"Refusing unrestricted execution: {probe.stderr[-2000:]}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def groundtruth_hashes(source: Path, task: str) -> dict[str, str]:
    gt_dir = source / "tasks/finalpool" / task / "groundtruth_workspace"
    if not gt_dir.is_dir():
        raise ValueError(f"Missing groundtruth_workspace for {task}: {gt_dir}")
    hashes = {path.name: _sha256(path) for path in sorted(gt_dir.iterdir()) if path.is_file()}
    if not hashes:
        raise ValueError(f"No groundtruth files found in {gt_dir}")
    return hashes


def build_quickstart_row(source: Path, task: str) -> dict[str, Any]:
    """Build a canonical dataset row from the pinned upstream Toolathlon checkout."""
    if task not in QUICKSTART_TASKS:
        raise ValueError(f"Unsupported quickstart task: {task}")
    task_dir = source / "tasks/finalpool" / task
    instruction_path = task_dir / "docs/task.md"
    config_path = task_dir / "task_config.json"
    eval_path = task_dir / "evaluation/main.py"
    if not instruction_path.is_file() or not config_path.is_file() or not eval_path.is_file():
        raise ValueError(f"Task {task} is missing required files in {task_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    instruction = instruction_path.read_text(encoding="utf-8").strip()
    gt_hashes = groundtruth_hashes(source, task)
    initial_dir = task_dir / "initial_workspace"
    initial_files = (
        sorted(path.name for path in initial_dir.iterdir() if path.is_file()) if initial_dir.is_dir() else []
    )
    return {
        "id": task,
        "input": {
            "task": task,
            "instruction": instruction,
            "mcp_servers": list(config.get("needed_mcp_servers", [])),
            "local_tools": list(config.get("needed_local_tools", [])),
            "initial_workspace_files": initial_files,
        },
        "expected": {
            "task": task,
            "upstream_revision": UPSTREAM_REVISION,
            "evaluation_module": f"tasks.finalpool.{task}.evaluation.main",
            "groundtruth_hashes": gt_hashes,
        },
        "metadata": {
            "runner": "native",
            "upstream_revision": UPSTREAM_REVISION,
        },
    }


def collect_workspace_files(workspace: Path, *, max_total_bytes: int = 4 * 1024 * 1024) -> dict[str, str]:
    """Transfer the complete workspace, or fail explicitly without a quality score.

    The raw-byte cap leaves room for base64 and JSON within Beaker's context
    wire limit. Never silently drop a file to fit it.
    """
    if workspace.is_symlink():
        raise ValueError("Workspace must not be a symlink")
    if not workspace.is_dir():
        return {}
    encoded: dict[str, str] = {}
    total_bytes = 0
    for path in sorted(workspace.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Workspace symlinks cannot be transferred: {path.relative_to(workspace)}")
        if not path.is_file():
            continue
        rel = path.relative_to(workspace).as_posix()
        remaining = max_total_bytes - total_bytes
        if path.stat().st_size > remaining:
            raise ValueError(f"Workspace exceeds the {max_total_bytes}-byte transfer limit at {rel!r}")
        with path.open("rb") as stream:
            data = stream.read(remaining + 1)
        total_bytes += len(data)
        if total_bytes > max_total_bytes:
            raise ValueError(f"Workspace exceeds the {max_total_bytes}-byte transfer limit at {rel!r}")
        encoded[rel] = base64.b64encode(data).decode("ascii")
    return encoded


def evaluate_workspace(
    *,
    source: Path,
    task: str,
    expected_hashes: dict[str, str],
    workspace_files: dict[str, str],
    agent_status: str | None,
    timeout: int = 300,
) -> tuple[bool, str]:
    """Run the official Toolathlon task evaluator in the trusted upstream environment."""
    if task not in QUICKSTART_TASKS:
        raise ValueError(f"Unsupported quickstart task: {task}")
    actual_hashes = groundtruth_hashes(source, task)
    if actual_hashes != expected_hashes:
        raise ValueError(f"Groundtruth hash mismatch for {task}")
    if agent_status != "success":
        return False, f"Agent status was {agent_status!r} (only 'success' is eligible to pass)."

    gt_dir = source / "tasks/finalpool" / task / "groundtruth_workspace"
    with tempfile.TemporaryDirectory(prefix=f"toolathlon-eval-{task}-") as temp_dir:
        root = Path(temp_dir)
        staged_gt_dir = root / "groundtruth_workspace"
        shutil.copytree(gt_dir, staged_gt_dir)
        workspace = root / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "arxiv_local_storage").mkdir(parents=True, exist_ok=True)
        for rel_path, b64_data in workspace_files.items():
            safe_rel = Path(rel_path)
            if safe_rel.is_absolute() or ".." in safe_rel.parts:
                raise ValueError(f"Invalid workspace path: {rel_path!r}")
            target = workspace / safe_rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(base64.b64decode(b64_data, validate=True))

        res_log_file = root / "traj_log.json"
        res_log_file.write_text(json.dumps({"status": agent_status}), encoding="utf-8")

        command = [
            str(source / ".venv/bin/python"),
            "-m",
            f"tasks.finalpool.{task}.evaluation.main",
            "--agent_workspace",
            str(workspace),
            "--groundtruth_workspace",
            str(staged_gt_dir),
            "--res_log_file",
            str(res_log_file),
        ]
        proc = subprocess.run(
            command,
            cwd=source,
            env=runtime_env(source),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        details = "\n".join(part.strip() for part in (proc.stdout, proc.stderr) if part and part.strip()).strip()
        if proc.returncode == 0:
            return True, details or "All evaluation checks passed."
        return False, details or f"Evaluation script exited with code {proc.returncode}."
