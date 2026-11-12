"""Stage and launch the native agent; only its system prompt is editable."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from toolathlon_baseline import UPSTREAM_REVISION
from toolathlon_baseline.prepare import native_path, prepared_source, require_command, runtime_env
from toolathlon_baseline.runner import stage_task


def stage_agent(source: Path, cache: Path, output: Path, task: str, system_prompt: Path | None) -> Path:
    checkout = stage_task(source, cache, output, task, system_prompt)
    task_dir = checkout / "tasks/finalpool" / task
    # Keep only solver inputs. References and evaluator code stay in the trusted
    # cache, which the Landlock policy does not grant the agent access to.
    for path in task_dir.iterdir():
        if path.name not in {"docs", "initial_workspace", "preprocess", "task_config.json"}:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    shutil.copyfile(Path(__file__).with_name("toolathlon_native_trace.py"), checkout / "toolathlon_native_trace.py")
    shutil.copyfile(Path(__file__).with_name("toolathlon_entry.py"), checkout / "agent_main.py")
    if "excel" in json.loads((task_dir / "task_config.json").read_text())["needed_mcp_servers"]:
        # This pinned server hardcodes its log beside site-packages. Relocate
        # its unmodified package so the log is per-case and the shared venv
        # remains read-only for the agent.
        package = next((source / ".venv-excel/lib").glob("python*/site-packages/excel_mcp"))
        shutil.copytree(package, checkout / "runtime/site-packages/excel_mcp")
    # MCP's stdio client forwards only a small environment allowlist. Supply
    # the per-case locations explicitly for every server and nested command.
    for path in (checkout / "configs/mcp_servers").glob("*.yaml"):
        config = yaml.safe_load(path.read_text())
        config["params"].setdefault("env", {}).update(
            HOME=str(output / "home"),
            TMPDIR=str(output / "tmp"),
            UV_CACHE_DIR=str(output / "uv-cache"),
            UV_PROJECT_ENVIRONMENT=str(source / ".venv"),
            UV_NO_SYNC="1",
            PYTHONPATH=f"{checkout / 'runtime/site-packages'}:{checkout}",
        )
        path.write_text(yaml.safe_dump(config, sort_keys=False))
    return checkout


def run_agent(
    *,
    cache: Path,
    output: Path,
    task: str,
    model: str,
    max_steps: int = 100,
    timeout: int = 1800,
    system_prompt: Path | None = None,
    model_params: Path | None = None,
) -> dict[str, Any]:
    if max_steps < 1 or timeout < 1:
        raise ValueError("max_steps and timeout must be positive")
    if not os.environ.get("TOOLATHLON_OPENAI_BASE_URL"):
        raise ValueError("Set TOOLATHLON_OPENAI_BASE_URL first")
    cache, output = native_path(cache), native_path(output)
    source = prepared_source(cache)
    output.mkdir(parents=True, exist_ok=False)
    checkout = stage_agent(source, cache, output, task, system_prompt)
    config = json.loads((source / "scripts/formal_run_v0.json").read_text())
    config["global_task_config"].update(dump_path=str(output / "artifacts"), direct_to_dumps=True)
    config["agent"]["model"] = {"short_name": model, "provider": "unified"}
    config_path = checkout / "eval_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    python = source / ".venv/bin/python"
    python_prefix = Path(
        subprocess.check_output([str(python), "-c", "import sys; print(sys.base_prefix)"], text=True).strip()
    )

    def execute() -> int:
        env = runtime_env(source)
        for name in ("home", "tmp", "uv-cache"):
            (output / name).mkdir()
        env.update(
            HOME=str(output / "home"),
            TMPDIR=str(output / "tmp"),
            UV_CACHE_DIR=str(output / "uv-cache"),
            UV_NO_SYNC="1",
            UV_PROJECT_ENVIRONMENT=str(source / ".venv"),
            PYTHONPATH=f"{checkout / 'runtime/site-packages'}:{checkout}",
            TOOLATHLON_TELEMETRY_STATUS=str(output / "telemetry_status.json"),
        )
        read_paths = [Path(p) for p in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc") if Path(p).exists()]
        read_paths += [
            source / ".venv",
            source / ".venv-excel",
            python_prefix,
            cache / "node",
            Path(require_command("node")),
        ]
        read_paths.append(Path(require_command("uv")))
        write_paths = [output, Path("/dev/null"), Path("/dev/urandom"), Path("/dev/random"), Path("/dev/zero")]
        for path in read_paths + write_paths:
            if source.resolve().is_relative_to(path.resolve()):
                raise ValueError(f"Sandbox allowlist would expose the trusted benchmark cache via {path}")
        policy_path = output / "sandbox.json"
        policy_path.write_text(json.dumps({"read": list(map(str, read_paths)), "write": list(map(str, write_paths))}))
        env["PYTHONUNBUFFERED"] = "1"
        env.pop("TOOLATHLON_MODEL_PARAMS_FILE", None)
        if model_params is not None:
            params_path = output / "model_params.json"
            params_path.write_bytes(model_params.read_bytes())
            env["TOOLATHLON_MODEL_PARAMS_FILE"] = str(params_path)
        launch = [
            str(python),
            str(Path(__file__).with_name("toolathlon_sandbox.py")),
            str(policy_path),
            str(python),
            "agent_main.py",
            "--agent_only",
            "--debug",
            "--eval_config",
            str(config_path),
            "--task_dir",
            f"finalpool/{task}",
            "--model_short_name",
            model,
            "--max_steps_under_single_turn_mode",
            str(max_steps),
        ]
        # The case supervisor owns this process group, including all MCPs.
        with (output / "run.log").open("w") as stream:
            return subprocess.run(
                launch, cwd=checkout, env=env, stdout=stream, stderr=stream, timeout=timeout
            ).returncode

    summary: dict[str, Any] = {
        "upstream_revision": UPSTREAM_REVISION,
        "task": f"finalpool/{task}",
        "model": model,
        "pass": None,
        "status": "error",
        "log": str(output / "run.log"),
    }
    try:
        summary["exit_code"] = execute()
        trajectory_path = output / "artifacts/traj_log.json"
        if trajectory_path.is_file():
            trajectory = json.loads(trajectory_path.read_text())
            summary["agent_status"] = trajectory.get("status")
            summary["key_stats"] = trajectory.get("key_stats")
            summary["status"] = "completed" if trajectory.get("status") == "success" else "failed"
    except subprocess.TimeoutExpired:
        summary["status"] = "timeout"
    finally:
        (output / "result.json").write_text(json.dumps(summary, indent=2))
    return summary
