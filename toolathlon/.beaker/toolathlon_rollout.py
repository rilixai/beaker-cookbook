"""Trusted native launch adapter; only the system prompt is candidate-editable."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from toolathlon_baseline import runner
from toolathlon_baseline.prepare import prepared_source, require_command


def stage_agent(source: Path, cache: Path, output: Path, task: str, system_prompt: Path | None) -> Path:
    checkout = _stage_task(source, cache, output, task, system_prompt)
    task_dir = checkout / "tasks/finalpool" / task
    # Keep only solver inputs. References and evaluator code stay in the trusted
    # cache, which the Landlock policy does not grant the agent access to.
    for path in task_dir.iterdir():
        if path.name not in {"docs", "initial_workspace", "preprocess", "task_config.json"}:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    shutil.copyfile(Path(__file__).with_name("toolathlon_agent.py"), checkout / "agent_main.py")
    if "excel" in json.loads((task_dir / "task_config.json").read_text())["needed_mcp_servers"]:
        # This pinned server hardcodes its log beside site-packages. Relocate
        # its unmodified package so the log is per-case and the shared venv
        # remains read-only for the agent.
        package = next((source / ".venv/lib").glob("python*/site-packages/excel_mcp"))
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


_stage_task = runner.stage_task


def run_agent(**kwargs: Any) -> dict[str, Any]:
    source = prepared_source(kwargs["cache"])
    python = source / ".venv/bin/python"
    python_prefix = Path(
        subprocess.check_output([str(python), "-c", "import sys; print(sys.base_prefix)"], text=True).strip()
    )

    def execute(command: list[str], *, cwd: Path, env: dict[str, str], log: Path, timeout: int) -> int:
        output = cwd.parent
        for name in ("home", "tmp", "uv-cache"):
            (output / name).mkdir()
        env.update(
            HOME=str(output / "home"),
            TMPDIR=str(output / "tmp"),
            UV_CACHE_DIR=str(output / "uv-cache"),
            UV_NO_SYNC="1",
            PYTHONPATH=f"{cwd / 'runtime/site-packages'}:{cwd}",
        )
        read_paths = [Path(p) for p in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc") if Path(p).exists()]
        read_paths += [source / ".venv", python_prefix, kwargs["cache"] / "node", Path(require_command("node"))]
        read_paths.append(Path(require_command("uv")))
        write_paths = [output, Path("/dev/null"), Path("/dev/urandom"), Path("/dev/random"), Path("/dev/zero")]
        for path in read_paths + write_paths:
            if source.resolve().is_relative_to(path.resolve()):
                raise ValueError(f"Sandbox allowlist would expose the trusted benchmark cache via {path}")
        policy_path = output / "sandbox.json"
        policy_path.write_text(json.dumps({"read": list(map(str, read_paths)), "write": list(map(str, write_paths))}))
        # Avoid a second process session: the onboarding supervisor owns the
        # entire group and can kill the agent and MCP servers on cancellation.
        main_index = command.index("main.py")
        launch = [
            str(python),
            str(Path(__file__).with_name("toolathlon_sandbox.py")),
            str(policy_path),
            str(python),
            "agent_main.py",
            *command[main_index + 1 :],
        ]
        with log.open("w") as stream:
            return subprocess.run(launch, cwd=cwd, env=env, stdout=stream, stderr=stream, timeout=timeout).returncode

    original_execute, original_stage = runner.execute, runner.stage_task
    try:
        runner.stage_task = stage_agent
        runner.execute = execute
        return runner.run_task(**kwargs)
    finally:
        runner.execute, runner.stage_task = original_execute, original_stage
