"""Run the original agent loop and evaluator in a fresh per-task source tree."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Any

import yaml

from toolathlon_baseline import QUICKSTART_TASKS, UPSTREAM_REVISION
from toolathlon_baseline.prepare import native_path, prepared_source, require_command, runtime_env


def execute(command: list[str], *, cwd: Path, env: dict[str, str], log: Path, timeout: int) -> int:
    """Bound the entire process group, including MCP servers, on timeout/Ctrl-C."""
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream, stderr=stream, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        finally:
            # A crashed parent may leave MCP descendants alive. Address the group
            # even when the original process has already exited.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            finally:
                # Descendants can outlive the original process or ignore TERM.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()


def stage_task(source: Path, cache: Path, output: Path, task: str, system_prompt: Path | None) -> Path:
    if task not in QUICKSTART_TASKS:
        raise ValueError("This task needs --environment PATH and the containerized runner; see the full setup guide")
    checkout = output / "source"
    checkout.mkdir()
    for directory in ("utils", "scripts"):
        shutil.copytree(
            source / directory, checkout / directory, ignore=shutil.ignore_patterns("__pycache__", "tests")
        )
    for name in ("main.py", "pyproject.toml", "uv.lock"):
        shutil.copyfile(source / name, checkout / name)
    task_dir = checkout / "tasks/finalpool" / task
    shutil.copytree(source / "tasks/finalpool" / task, task_dir)
    if system_prompt is not None:
        (task_dir / "docs/agent_system_prompt.md").write_text(system_prompt.read_text())
    configs = checkout / "configs"
    configs.mkdir()
    for name in ("global_configs", "token_key_session"):
        shutil.copyfile(source / f"configs/{name}_example.py", configs / f"{name}.py")

    needed = json.loads((task_dir / "task_config.json").read_text())["needed_mcp_servers"]
    servers = configs / "mcp_servers"
    servers.mkdir()
    found = set()
    bin_dir = source / ".venv/bin"
    entrypoints = {
        "scholarly": "mcp-scholarly",
        "word": "word_mcp_server",
        "excel": "excel-mcp-server",
        "terminal": "cli-mcp-server",
    }
    for path in (source / "configs/mcp_servers").glob("*.yaml"):
        config = yaml.safe_load(path.read_text())
        name = config.get("name", path.stem)
        if name not in needed:
            continue
        params = config["params"]
        # Only launch-path adaptations: all tools and implementations are upstream.
        # Direct executables avoid uvx/npx resolving newer versions at run time.
        if name == "filesystem":
            params["command"] = require_command("node")
            params["args"] = [
                str(cache / "node/node_modules/@modelcontextprotocol/server-filesystem/dist/index.js"),
                "${agent_workspace}",
            ]
        elif name == "arxiv_local":
            params["command"] = str(bin_dir / "python")
            params["args"] = [
                "-m",
                "utils.local_servers.arxiv_local_wrapper",
                "--storage-path",
                "${agent_workspace}/arxiv_local_storage",
            ]
            params.setdefault("env", {})["PYTHONPATH"] = str(checkout)
        else:
            params["command"] = str(bin_dir / entrypoints[name])
            params["args"] = ["stdio"] if name == "excel" else []
        (servers / path.name).write_text(yaml.safe_dump(config, sort_keys=False))
        found.add(name)
    if found != set(needed):
        raise RuntimeError(f"Missing MCP server definitions: {set(needed) - found}")
    return checkout


def read_result(output: Path) -> dict[str, Any]:
    summary: dict[str, Any] = json.loads((output / "result.json").read_text())
    return summary


def check_tools(*, cache: Path, output: Path, task: str, timeout: int = 180) -> dict[str, Any]:
    """Connect to the actual task tools without sending a model request."""
    cache = native_path(cache)
    source = prepared_source(cache)
    output = native_path(output)
    output.mkdir(parents=True, exist_ok=False)
    checkout = stage_task(source, cache, output, task, None)
    workspace = output / "artifacts/workspace"
    workspace.mkdir(parents=True)
    (workspace / "arxiv_local_storage").mkdir()
    script = """
import asyncio, json, sys
from pathlib import Path
from utils.mcp.tool_servers import MCPServerManager

async def main():
    workspace, task, report = sys.argv[1:]
    names = json.loads(Path(f'tasks/finalpool/{task}/task_config.json').read_text())['needed_mcp_servers']
    manager = MCPServerManager(workspace)
    try:
        await manager.connect_servers(names)
        tools = {}
        for name in names:
            tools[name] = [tool.name for tool in await manager.connected_servers[name].list_tools()]
            if not tools[name]:
                raise RuntimeError(f'{name} returned no tools')
        Path(report).write_text(json.dumps(tools, indent=2) + '\\n')
    finally:
        await manager.ensure_all_disconnected()

asyncio.run(main())
"""
    try:
        code = execute(
            [str(source / ".venv/bin/python"), "-c", script, str(workspace), task, str(output / "tools.json")],
            cwd=checkout,
            env=runtime_env(source),
            log=output / "check.log",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"MCP check timed out; see {output / 'check.log'}") from exc
    if code != 0:
        raise RuntimeError(f"MCP check failed; see {output / 'check.log'}")
    tools: dict[str, list[str]] = json.loads((output / "tools.json").read_text())
    return {"task": task, "tools": {name: len(entries) for name, entries in tools.items()}}


def run_task(
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
        raise ValueError("Set TOOLATHLON_OPENAI_BASE_URL to your OpenAI-compatible endpoint first")
    cache = native_path(cache)
    source = prepared_source(cache)
    output = native_path(output)
    # Never overwrite trajectories or reuse a previous task's application state.
    output.mkdir(parents=True, exist_ok=False)
    checkout = stage_task(source, cache, output, task, system_prompt)
    config = json.loads((source / "scripts/formal_run_v0.json").read_text())
    config["global_task_config"].update(dump_path=str(output / "artifacts"), direct_to_dumps=True)
    config["agent"]["model"] = {"short_name": model, "provider": "unified"}
    config["user"]["model"] = {"short_name": model, "provider": "unified"}
    config_path = checkout / "eval_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    env = runtime_env(source)
    env["PYTHONUNBUFFERED"] = "1"
    # Native subprocesses must not inherit a different model-parameter file.
    env.pop("TOOLATHLON_MODEL_PARAMS_FILE", None)
    if model_params is not None:
        params = json.loads(model_params.read_text())
        if not isinstance(params, dict):
            raise ValueError("The model parameters file must contain a JSON object")
        params_path = output / "model_params.json"
        params_path.write_text(json.dumps(params, indent=2) + "\n")
        env["TOOLATHLON_MODEL_PARAMS_FILE"] = str(params_path)
    command = [
        require_command("uv"),
        "run",
        "--frozen",
        "main.py",
        "--eval_config",
        str(config_path),
        "--task_dir",
        f"finalpool/{task}",
        "--model_short_name",
        model,
        "--provider",
        "unified",
        "--max_steps_under_single_turn_mode",
        str(max_steps),
    ]
    summary: dict[str, Any] = {
        "upstream_revision": UPSTREAM_REVISION,
        "task": f"finalpool/{task}",
        "model": model,
        "pass": None,
        "status": "error",
        "log": str(output / "run.log"),
    }
    try:
        summary["exit_code"] = execute(command, cwd=checkout, env=env, log=output / "run.log", timeout=timeout)
        evaluation_path = output / "artifacts/eval_res.json"
        if evaluation_path.is_file():
            evaluation = json.loads(evaluation_path.read_text())
            verdict = evaluation.get("pass")
            if verdict is not None and type(verdict) is not bool:
                raise ValueError("Upstream returned an invalid pass/fail verdict")
            summary["evaluation"] = evaluation
            summary["pass"] = verdict
            if verdict is False:
                summary["status"] = "failed"
            elif verdict is True and summary["exit_code"] == 0:
                summary["status"] = "passed"
        trajectory_path = output / "artifacts/traj_log.json"
        if trajectory_path.is_file():
            trajectory = json.loads(trajectory_path.read_text())
            summary["agent_status"] = trajectory.get("status")
            summary["key_stats"] = trajectory.get("key_stats")
            summary["agent_cost"] = trajectory.get("agent_cost")
    except subprocess.TimeoutExpired:
        summary["status"] = "timeout"
    finally:
        (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
