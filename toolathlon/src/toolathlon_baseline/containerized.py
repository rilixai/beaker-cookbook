"""Run any official task using Toolathlon's configured Linux environment."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from toolathlon_baseline import UPSTREAM_REVISION
from toolathlon_baseline.catalog import select_tasks, task_catalog
from toolathlon_baseline.prepare import native_path, require_command, runtime_env, verify_source
from toolathlon_baseline.runner import execute


DEFAULT_IMAGE = "lockon0927/toolathlon-task-image:1016beta"


def configured_source(environment: Path) -> Path:
    source = native_path(environment)
    # Full setup changes configuration and task-specific resource IDs. It must
    # retain the pinned Git revision, but cannot be required to remain clean.
    verify_source(source, require_clean=False)
    return source


def check_environment(environment: Path, image: str = DEFAULT_IMAGE) -> dict[str, Any]:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("The full containerized benchmark requires a Linux host")
    source = configured_source(environment)
    for name in (
        ".venv/bin/python",
        "configs/global_configs.py",
        "configs/token_key_session.py",
        "configs/gcp-oauth.keys.json",
        "configs/google_credentials.json",
    ):
        if not (source / name).is_file():
            raise RuntimeError(f"Missing {name}; complete the upstream full-environment setup in {source}")
    # Only emit the runtime selection, never the configuration or credentials.
    runtime = subprocess.check_output(
        [
            str(source / ".venv/bin/python"),
            "-c",
            "from configs.global_configs import global_configs; print(global_configs.podman_or_docker)",
        ],
        cwd=source,
        env=runtime_env(source),
        text=True,
    ).strip()
    if runtime not in ("docker", "podman"):
        raise ValueError("configs/global_configs.py must select docker or podman")
    require_command(runtime)
    require_command("uv")
    if not re.fullmatch(r"[A-Za-z0-9_./:@+-]+", image):
        raise ValueError("Unsupported image name")
    for args in ([runtime, "info"], [runtime, "image", "inspect", image]):
        completed = subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if completed.returncode:
            raise RuntimeError(f"Container runtime or image unavailable: {' '.join(args)}; see the full setup guide")
    return {
        "environment": str(source),
        "runtime": runtime,
        "image": image,
        "image_id": subprocess.check_output(
            [runtime, "image", "inspect", "--format", "{{.Id}}", image],
            text=True,
        ).strip(),
        "tasks": len(task_catalog(source)),
        "upstream_revision": UPSTREAM_REVISION,
        "note": "Runtime/configuration preflight only; run upstream MCP installation verification separately",
    }


def cleanup_task_containers(runtime: str, artifacts: Path) -> None:
    """Remove only benchmark containers whose bind mounts belong to this run."""
    containers = subprocess.check_output(
        [runtime, "ps", "--all", "--quiet", "--filter", "name=toolathlon-"],
        text=True,
        timeout=30,
    ).split()
    root = artifacts.resolve()
    for container in containers:
        # Inspect mount metadata alone; full inspect output contains credentials.
        inspected = subprocess.run(
            [runtime, "inspect", "--format", "{{json .Mounts}}", container],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if inspected.returncode:
            continue  # The official EXIT trap may already have removed it.
        mounts = json.loads(inspected.stdout)
        belongs_to_run = any(
            mount.get("Source") and Path(mount["Source"]).resolve().is_relative_to(root) for mount in mounts
        )
        if belongs_to_run:
            subprocess.run([runtime, "rm", "--force", container], check=True, capture_output=True, timeout=30)


def collect_results(artifacts: Path, tasks: list[str]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for task in tasks:
        folder = artifacts / "finalpool" / task
        result: dict[str, Any] = {"pass": None, "status": "error", "artifacts": str(folder)}
        try:
            evaluation = json.loads((folder / "eval_res.json").read_text())
            verdict = evaluation.get("pass")
            if verdict is not None and type(verdict) is not bool:
                raise ValueError("Invalid upstream verdict")
            result["evaluation"] = evaluation
            result["pass"] = verdict
            if verdict is not None and evaluation.get("failure") not in ("evaluation_error", "log_file_not_found"):
                result["status"] = "passed" if verdict else "failed"
        except (OSError, ValueError, AttributeError) as exc:
            result["error"] = type(exc).__name__
        results[task] = result
    return results


def run_containerized(
    *,
    environment: Path,
    output: Path,
    tasks: list[str] | None,
    model: str,
    workers: int = 1,
    max_steps: int = 100,
    timeout: int = 5400,
    image: str = DEFAULT_IMAGE,
    model_params: Path | None = None,
) -> dict[str, Any]:
    if min(workers, max_steps, timeout) < 1:
        raise ValueError("workers, max_steps, and timeout must be positive")
    if not os.environ.get("TOOLATHLON_OPENAI_BASE_URL"):
        raise ValueError("Set TOOLATHLON_OPENAI_BASE_URL first")
    # Upstream's scheduler constructs a shell command from the model name.
    if not re.fullmatch(r"[A-Za-z0-9_./:@+-]+", model):
        raise ValueError("Containerized model IDs cannot contain spaces or shell metacharacters")
    source = configured_source(environment)
    selected = select_tasks(source, tasks)
    preflight = check_environment(source, image)
    output = native_path(output)
    output.mkdir(parents=True, exist_ok=False)
    artifacts = output / "artifacts"
    artifacts.mkdir()
    config = json.loads((source / "scripts/formal_run_v0.json").read_text())
    config["global_task_config"].update(dump_path="/workspace/dumps", direct_to_dumps=True)
    for role in ("agent", "user"):
        config[role]["model"] = {"short_name": model, "provider": "unified"}
    (output / "eval_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "tasks.txt").write_text("\n".join(selected) + "\n")
    env = runtime_env(source)
    env["PYTHONUNBUFFERED"] = "1"
    env["TOOLATHLON_CONTAINERIZED_MODE"] = "phased"
    env.pop("TOOLATHLON_MODEL_PARAMS_FILE", None)
    if model_params is not None:
        params = json.loads(model_params.read_text())
        if not isinstance(params, dict):
            raise ValueError("The model parameters file must contain a JSON object")
        params_file = output / "model_params.json"
        params_file.write_text(json.dumps(params, indent=2) + "\n")
        env["TOOLATHLON_MODEL_PARAMS_FILE"] = str(params_file)
    summary: dict[str, Any] = {
        "upstream_revision": UPSTREAM_REVISION,
        "runner": "containerized",
        "image": image,
        "image_id": preflight["image_id"],
        "model": model,
        "status": "error",
        "requested_tasks": selected,
        "workers": workers,
        "timeout_per_task": timeout,
        "log": str(output / "run.log"),
    }
    temp_root = source / "scripts/temp_configs"
    temp_root.mkdir(exist_ok=True)
    interrupted = False
    try:
        # The script directory is copied into each container. Its config path
        # must be relative to the upstream checkout to exist under /workspace.
        with tempfile.TemporaryDirectory(prefix="cookbook-", dir=temp_root) as tmp:
            config_path = Path(tmp) / "eval_config.json"
            config_path.write_text(json.dumps(config))
            command = [
                require_command("uv"),
                "run",
                "--frozen",
                "run_parallel.py",
                "--tasks_folder",
                "finalpool",
                "--tag",
                "normal",
                "--model_short_name",
                model,
                "--provider",
                "unified",
                "--maxstep",
                str(max_steps),
                "--workers",
                str(workers),
                "--timeout",
                str(timeout),
                "--dump_path",
                str(artifacts),
                "--task_list",
                str(output / "tasks.txt"),
                "--eval_config",
                str(config_path.relative_to(source)),
                "--image_name",
                image,
                "--runner",
                "containerized",
            ]
            # This is a last-resort global ceiling. The official scheduler owns
            # per-task deadlines, resource conflict locks, and task scheduling.
            ceiling = len(selected) * (timeout + 120) + 300
            summary["exit_code"] = execute(command, cwd=source, env=env, log=output / "run.log", timeout=ceiling)
    except subprocess.TimeoutExpired:
        interrupted = True
        summary["status"] = "timeout"
    except KeyboardInterrupt:
        interrupted = True
        summary["status"] = "interrupted"
        raise
    finally:
        try:
            cleanup_task_containers(preflight["runtime"], artifacts)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            summary["cleanup_error"] = type(exc).__name__
        results = collect_results(artifacts, selected)
        counts = {
            status: sum(result["status"] == status for result in results.values())
            for status in ("passed", "failed", "error")
        }
        summary.update(
            tasks=results,
            counts={"total": len(selected), **counts},
            pass_rate=counts["passed"] / len(selected) if counts["error"] == 0 else None,
        )
        if not interrupted:
            if summary.get("exit_code") == 0 and counts["error"] == 0 and "cleanup_error" not in summary:
                summary["status"] = "passed" if counts["failed"] == 0 else "failed"
        (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
