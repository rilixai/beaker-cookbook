"""Run one task in an existing, fully provisioned ALE installation."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml


async def run_task(task_id: str, *, commit: str) -> dict[str, Any]:
    checkout = Path(os.environ["ALE_CHECKOUT_PATH"]).resolve()
    environment = Path(os.environ["ALE_ENVIRONMENT_CONFIG"]).resolve()
    actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit:
        raise ValueError("ALE checkout does not match the reviewed task commit")
    subprocess.run(["git", "-C", str(checkout), "diff", "--exit-code", "HEAD", "--", "tasks", "ale_run"], check=True)
    if not environment.is_file():
        raise FileNotFoundError("ALE_ENVIRONMENT_CONFIG must name a provisioned upstream environment config")
    with tempfile.TemporaryDirectory(prefix="beaker-ale-case-") as temporary:
        root = Path(temporary)
        tasks = root / "tasks.txt"
        tasks.write_text(task_id + "\n")
        logs = root / "logs"
        experiment = root / "experiment.yaml"
        experiment.write_text(
            yaml.safe_dump(
                {
                    "name": "beaker-case",
                    "agents": [str(Path(__file__).parents[1] / "src/agents_last_exam/agent.yaml")],
                    "environment": str(environment),
                    "tasks": str(tasks),
                    "output": {"root": str(logs)},
                    "concurrency": 1,
                    "auto_resume": False,
                    "max_attempts": 1,
                    "cleanup_mode": "delete",
                }
            )
        )
        with (root / "runner.log").open("wb") as output:
            process = await asyncio.create_subprocess_exec(
                "uv",
                "run",
                "--locked",
                "--all-packages",
                "python",
                "-m",
                "ale_run",
                "run",
                str(experiment),
                cwd=checkout,
                stdout=output,
                stderr=output,
                start_new_session=True,
            )
            try:
                code = await process.wait()
            except BaseException:
                if process.returncode is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        await asyncio.wait_for(process.wait(), 30)
                    except TimeoutError:
                        os.killpg(process.pid, signal.SIGKILL)
                        await process.wait()
                raise
        if code:
            raise RuntimeError(f"ALE runner failed (exit {code}); inspect the configured sandbox and credentials")
        results = sorted(logs.rglob("eval_result.json"))
        if len(results) != 1:
            raise ValueError(f"Expected one ALE variant result, got {len(results)}")
        result: dict[str, Any] = json.loads(results[0].read_text())
        if result.get("eval_status") != "success" or result.get("error"):
            raise RuntimeError("ALE evaluation failed; infrastructure failures are not task score zero")
        return {"task_id": task_id, "score": result["score"], "eval_status": result["eval_status"]}
