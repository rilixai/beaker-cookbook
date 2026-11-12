"""Beaker repository Integration for the native Toolathlon benchmark."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import tempfile
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from beaker import (
    Case,
    CaseResult,
    CaseScore,
    Check,
    Integration,
    JsonValue,
    RepositoryRunSetupResult,
    RetryableCaseError,
    RolloutRuntime,
    RunSetup,
    SetupRuntime,
    inference_target,
    repository,
)
from pydantic import BaseModel, Field
from toolathlon_agent_runtime import prepare_agent_runtime
from toolathlon_native_trace import NativeTrace
from toolathlon_setup import (
    build_quickstart_row,
    collect_workspace_files,
    evaluate_workspace,
    prepare_toolathlon,
    verify_sandbox_support,
)

from toolathlon_baseline import QUICKSTART_TASKS, UPSTREAM_REVISION


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-6-luna"
DEFAULT_MAX_STEPS = 100
DEFAULT_TIMEOUT_SECONDS = 1800

_SUBPROCESS_SCRIPT = """
import json
import pathlib
import sys

from toolathlon_rollout import run_agent

request = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
model_params = pathlib.Path(request["model_params"]) if request.get("model_params") else None
summary = run_agent(
    cache=pathlib.Path(request["cache"]),
    output=pathlib.Path(request["output"]),
    task=request["task"],
    model=request["model"],
    max_steps=int(request["max_steps"]),
    timeout=int(request["timeout"]),
    model_params=model_params,
    system_prompt=pathlib.Path(request["system_prompt"]),
    agent_venv=pathlib.Path(request["agent_venv"]),
)
pathlib.Path(sys.argv[2]).write_text(json.dumps(summary), encoding="utf-8")
"""


class TaskInput(BaseModel):
    task: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    mcp_servers: list[str] = Field(min_length=1)
    local_tools: list[str] = Field(default_factory=list)
    initial_workspace_files: list[str] = Field(default_factory=list)


class TaskExpected(BaseModel):
    task: str = Field(min_length=1)
    upstream_revision: str = Field(min_length=1)
    evaluation_module: str = Field(min_length=1)
    groundtruth_hashes: dict[str, str] = Field(min_length=1)


class Row(BaseModel):
    id: str = Field(min_length=1)
    input: TaskInput
    expected: TaskExpected
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class Setup(RunSetup[Row]):
    row_model = Row

    @asynccontextmanager
    async def prepare_run(self, *, runtime: SetupRuntime):
        del runtime
        verify_sandbox_support()
        self.cache, self.source = prepare_toolathlon()
        prepare_agent_runtime(self.cache, self.source)
        yield {"repository": RepositoryRunSetupResult()}

    async def load_cases(self, row: Row, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        del runtime
        if row.id not in QUICKSTART_TASKS or row.input.task != row.id or row.expected.task != row.id:
            raise ValueError(f"Unsupported or mismatched Toolathlon quickstart task id: {row.id!r}")
        canonical = build_quickstart_row(self.source, row.id)
        if row.input.instruction != canonical["input"]["instruction"]:
            raise ValueError(f"Instruction mismatch for task {row.id!r}")
        if row.expected.model_dump() != canonical["expected"]:
            raise ValueError(f"Expected ground-truth specification mismatch for task {row.id!r}")
        yield Case(
            id=row.id,
            input=row.input.model_dump(),
            expected=row.expected.model_dump(),
            metadata=row.metadata,
        )


def _resolve_model_target(runtime: RolloutRuntime[Any]) -> tuple[str, str, str, bool]:
    """Return (base_url, api_key, model_name, is_gateway) for a case rollout."""
    if runtime.model:
        target = inference_target(runtime)
        return target.base_url, target.api_key, target.model, True
    base_url = os.environ.get("TOOLATHLON_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    if "openrouter.ai" in base_url:
        api_key = os.environ.get("TOOLATHLON_OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or "fake-key"
    else:
        api_key = (
            os.environ.get("TOOLATHLON_OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
            or "fake-key"
        )
    model = os.environ.get("TOOLATHLON_MODEL") or DEFAULT_MODEL
    return base_url, api_key, model, False


async def _communicate(process: asyncio.subprocess.Process) -> tuple[bytes, bytes]:
    """Bound the supervisor and reap its process group even on cancellation."""
    try:
        return await asyncio.wait_for(process.communicate(), timeout=DEFAULT_TIMEOUT_SECONDS + 30)
    finally:
        alive = False
        try:
            os.killpg(process.pid, signal.SIGTERM)
            alive = True
        except ProcessLookupError:
            pass
        # Give tools a short opportunity to disconnect, then kill descendants
        # even if the supervisor has already exited or ignores SIGTERM.
        try:
            if alive:
                await asyncio.sleep(5)
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()


def _has_model_turn(agent_status: Any, key_stats: Any, trajectory: dict[str, Any]) -> bool:
    # Upstream only finalizes agent_llm_requests after a successful Runner.run.
    # MaxTurnsExceeded can leave it at zero after many actual model/tool turns.
    if agent_status == "max_turns_reached":
        return True
    if isinstance(key_stats, Mapping) and int(key_stats.get("agent_llm_requests") or 0) > 0:
        return True
    return any(
        isinstance(message, Mapping) and message.get("role") in {"assistant", "tool"}
        for message in trajectory.get("messages", [])
    )


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime[Any]) -> CaseResult:
    payload = TaskInput.model_validate(case_input)
    task = payload.task
    cache, source = prepare_toolathlon()
    agent_venv = prepare_agent_runtime(cache, source)
    base_url, api_key, model_name, is_gateway = _resolve_model_target(runtime)

    with (
        NativeTrace(runtime.trace) as telemetry,
        runtime.trace.stage(
            "toolathlon.run_task",
            inputs={"task": task, "model": model_name, "max_steps": DEFAULT_MAX_STEPS},
        ) as stage,
    ):
        with tempfile.TemporaryDirectory(prefix=f"beaker-toolathlon-case-{task}-") as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "run"
            telemetry.status_path = output_dir / "telemetry_status.json"
            request_path = root / "request.json"
            response_path = root / "response.json"

            request_payload: dict[str, Any] = {
                "cache": str(cache),
                "output": str(output_dir),
                "task": task,
                "model": model_name,
                "max_steps": DEFAULT_MAX_STEPS,
                "timeout": DEFAULT_TIMEOUT_SECONDS,
                "system_prompt": str(Path(__file__).resolve().parents[1] / "prompts/agent_system_prompt.md"),
                "agent_venv": str(agent_venv),
            }
            custom_params_file = os.environ.get("TOOLATHLON_MODEL_PARAMS_FILE")
            if custom_params_file and not is_gateway and Path(custom_params_file).is_file():
                request_payload["model_params"] = custom_params_file
            else:
                params_file = root / "gateway_model_params.json"
                params_file.write_text(json.dumps({"tool_choice": "auto"}), encoding="utf-8")
                request_payload["model_params"] = str(params_file)
            request_path.write_text(json.dumps(request_payload), encoding="utf-8")

            env = os.environ.copy()
            env["TOOLATHLON_OPENAI_BASE_URL"] = base_url
            env["TOOLATHLON_OPENAI_API_KEY"] = api_key
            env.update(telemetry.env())
            project_root = Path(__file__).resolve().parents[1]
            src_dir = os.pathsep.join((str(project_root / ".beaker"), str(project_root / "src")))
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing_pythonpath}".rstrip(os.pathsep)

            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                _SUBPROCESS_SCRIPT,
                str(request_path),
                str(response_path),
                cwd=str(Path(__file__).resolve().parents[1]),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout_bytes, stderr_bytes = await _communicate(process)
            except TimeoutError as exc:
                raise RetryableCaseError(f"Toolathlon supervisor timed out for {task!r}") from exc
            finally:
                telemetry.read_status()
            if process.returncode != 0 or not response_path.is_file():
                detail = (stderr_bytes or stdout_bytes).decode("utf-8", errors="replace")[-4000:]
                raise RetryableCaseError(f"Toolathlon runner subprocess failed (exit {process.returncode}): {detail}")

            summary = json.loads(response_path.read_text(encoding="utf-8"))
            traj_path = output_dir / "artifacts/traj_log.json"
            trajectory: dict[str, Any] = (
                json.loads(traj_path.read_text(encoding="utf-8")) if traj_path.is_file() else {}
            )

            status = summary.get("status")
            agent_status = summary.get("agent_status") or trajectory.get("status")
            key_stats = summary.get("key_stats") or trajectory.get("key_stats") or {}
            llm_requests = int(key_stats.get("agent_llm_requests") or 0) if isinstance(key_stats, Mapping) else 0

            if status == "timeout" or agent_status == "timeout":
                stage.output({"status": "timeout", "llm_requests": llm_requests})
                raise RetryableCaseError(f"Toolathlon task {task!r} timed out after {DEFAULT_TIMEOUT_SECONDS}s")

            if agent_status != "success" and not _has_model_turn(agent_status, key_stats, trajectory):
                log_path = output_dir / "run.log"
                log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:] if log_path.is_file() else ""
                stage.output({"status": "error_before_first_turn", "log_tail": log_tail})
                raise RetryableCaseError(f"Toolathlon task {task!r} exited before completing a model turn: {log_tail}")

            workspace_dir = output_dir / "artifacts/workspace"
            try:
                workspace_files = collect_workspace_files(workspace_dir) if agent_status == "success" else {}
            except (ValueError, OSError) as exc:
                raise RetryableCaseError(f"Cannot transfer the complete workspace for {task!r}: {exc}") from exc

            messages = trajectory.get("messages") if isinstance(trajectory.get("messages"), list) else []
            final_message = next(
                (
                    msg.get("content")
                    for msg in reversed(messages)
                    if isinstance(msg, Mapping) and msg.get("role") == "assistant" and msg.get("content")
                ),
                None,
            )

            observed_output: dict[str, JsonValue] = {
                "task": task,
                "upstream_revision": UPSTREAM_REVISION,
                "agent_framework": "openai-agents==0.20.0",
                "trace_source": "native_otlp",
                "model": model_name,
                "agent_status": agent_status,
                "key_stats": key_stats if isinstance(key_stats, Mapping) else {},
                "workspace_files": sorted(workspace_files.keys()),
                "final_assistant_message": final_message,
            }
            stage.output(
                {
                    "agent_status": agent_status,
                    "llm_requests": llm_requests,
                    "workspace_file_count": len(workspace_files),
                }
            )
            context: dict[str, JsonValue] = {
                "task": task,
                "agent_status": agent_status,
                "workspace_files": workspace_files,
            }
            return CaseResult(output=observed_output, output_kind="record", context=context)


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    del case_files_dir
    expected = TaskExpected.model_validate(case.expected)
    _, source = prepare_toolathlon()

    context = result.context if isinstance(result.context, Mapping) else {}
    if context.get("task") != expected.task or not isinstance(context.get("agent_status"), str):
        raise RetryableCaseError("Missing or mismatched Toolathlon scoring context; cannot assign a quality score")
    agent_status = str(context.get("agent_status") or "") or None
    raw_files = context.get("workspace_files")
    if not isinstance(raw_files, Mapping) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in raw_files.items()
    ):
        raise RetryableCaseError("Missing or malformed Toolathlon workspace transfer; cannot assign a quality score")
    workspace_files = dict(raw_files)

    passed, details = await asyncio.to_thread(
        evaluate_workspace,
        source=source,
        task=expected.task,
        expected_hashes=expected.groundtruth_hashes,
        workspace_files=workspace_files,
        agent_status=agent_status,
    )
    score_value = 1.0 if passed else 0.0
    check = Check(
        name=f"{expected.task}.official_evaluation",
        group=expected.task,
        verdict="pass" if passed else "fail",
        message=None if passed else details,
        description=f"Official Toolathlon evaluator ({expected.evaluation_module}) against groundtruth_workspace.",
    )
    return CaseScore(
        objective=score_value,
        field_scores={"pass_rate": score_value},
        checks=(check,),
    )


integration = Integration(
    targets=repository(("prompts/agent_system_prompt.md",)),
    run_setup=Setup,
    run_case=run_case,
    score_case=score_case,
)
