"""Optimize the Toolathlon default harness on every Toolathlon task.

Each case runs Toolathlon's decoupled mode: the task container (started with
Docker inside the rollout sandbox) prepares the environment, serves the MCP
tools over one SSE gateway, and grades the result, while the candidate's
``toolathlon_default`` agent loop runs in the sandbox against Beaker's model.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import tempfile
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from beaker import (
    Case,
    CaseResult,
    CaseScore,
    Check,
    Integration,
    RepositoryRunSetupResult,
    RunSetup,
    inference_target,
    repository,
)
from pydantic import BaseModel, Field

from toolathlon_setup import TASK_IMAGE, TASKS, UPSTREAM, prepare_task, prepare_toolathlon, required_env, tool_env


# Toolathlon's own per-task limits (scripts/formal_run_v0.json caps turns at 50).
MAX_STEPS = int(os.environ.get("TOOLATHLON_MAX_STEPS", "100"))
CASE_TIMEOUT_SECONDS = int(os.environ.get("TOOLATHLON_CASE_TIMEOUT_SECONDS", "5400"))
_LOG_TAIL_CHARS = 4000
_LOG_FILES = ("preprocess.log", "host_loop.log", "eval.log", "container.log")


class TaskInput(BaseModel):
    task: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    prompt: str = Field(min_length=1)


class Row(BaseModel):
    id: str
    input: TaskInput
    expected: dict[str, Any] = Field(default_factory=dict)


class Setup(RunSetup[Row]):
    row_model = Row

    @asynccontextmanager
    async def prepare_run(
        self, *, runtime: Any
    ) -> AsyncIterator[Mapping[Literal["repository", "documents"], RepositoryRunSetupResult]]:
        await asyncio.to_thread(prepare_toolathlon)
        yield {"repository": RepositoryRunSetupResult()}

    async def load_cases(self, row: Row, *, runtime: Any) -> AsyncIterator[Case]:
        task_dir = TASKS / row.input.task
        if row.id != row.input.task or not task_dir.is_dir():
            raise ValueError(f"Unknown Toolathlon task {row.input.task!r}")
        if row.input.prompt != (task_dir / "docs" / "task.md").read_text():
            raise ValueError(f"Prompt mismatch for {row.input.task}")
        missing = [name for name in required_env(row.input.task) if not os.environ.get(name)]
        if missing:
            raise ValueError(f"Task {row.input.task} needs platform secrets: {', '.join(missing)}")
        yield Case(id=row.id, input=row.input.model_dump(), expected=row.expected)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _tail(path: Path) -> str:
    try:
        return path.read_text(errors="replace")[-_LOG_TAIL_CHARS:]
    except OSError:
        return ""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.is_file() else None


async def run_case(*, case_input: Any, runtime: Any) -> CaseResult:
    task = TaskInput.model_validate(case_input)
    target = inference_target(runtime)
    await asyncio.to_thread(prepare_task, task.task)
    env = tool_env() | {
        "TOOLATHLON_OPENAI_BASE_URL": target.base_url,
        "TOOLATHLON_OPENAI_API_KEY": target.api_key,
    }
    with (
        tempfile.TemporaryDirectory(
            prefix="toolathlon-case-",
            dir=os.environ.get("TOOLATHLON_DUMP_ROOT"),
            delete=not os.environ.get("TOOLATHLON_DUMP_ROOT"),
        ) as dump,
        runtime.trace.stage("toolathlon.run_case", inputs={"task": task.task, "model": target.model}) as stage,
    ):
        process = await asyncio.create_subprocess_exec(
            "bash",
            "scripts/run_single_decoupled.sh",
            f"finalpool/{task.task}",
            "normal",
            dump,
            target.model,
            "unified",
            str(MAX_STEPS),
            "scripts/formal_run_v0.json",
            TASK_IMAGE,
            "toolathlon_default",
            str(_free_port()),
            cwd=UPSTREAM,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=CASE_TIMEOUT_SECONDS)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise
        folder = Path(dump) / "finalpool" / task.task
        evaluation = next((_read_json(path) for path in sorted(folder.rglob("eval_res.json"))), None)
        logs = {name: _tail(folder / name) for name in _LOG_FILES}
        if not isinstance(evaluation, dict):
            raise RuntimeError(
                f"Toolathlon produced no evaluation for {task.task} (exit {process.returncode}):\n"
                + stdout.decode(errors="replace")[-_LOG_TAIL_CHARS:]
                + "".join(f"\n--- {name}\n{text}" for name, text in logs.items() if text)
            )
        stage.output({"pass": evaluation.get("pass"), "exit_code": process.returncode})
        return CaseResult(
            output={
                "pass": evaluation.get("pass"),
                "details": str(evaluation.get("details") or evaluation.get("failure") or "")[-_LOG_TAIL_CHARS:],
                "exit_code": process.returncode,
                "host_loop_log": logs["host_loop.log"],
            },
            output_kind="record",
        )


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    del case, case_files_dir
    if not isinstance(result.output, dict):
        raise ValueError("Toolathlon output must be the evaluation record")
    passed = result.output.get("pass") is True
    return CaseScore(
        objective=float(passed),
        field_scores={"pass": float(passed)},
        checks=(
            Check(
                name="toolathlon_evaluation",
                verdict="pass" if passed else "fail",
                message=str(result.output.get("details") or "") or None,
            ),
        ),
    )


integration = Integration(
    targets=repository(
        (
            "upstream/utils/roles",
            "upstream/utils/aux_tools",
            "upstream/utils/system_prompts",
            "upstream/scripts/decoupled/host_agent_loop.py",
        )
    ),
    run_setup=Setup,
    run_case=run_case,
    score_case=score_case,
)
