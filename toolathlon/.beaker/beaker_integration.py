"""Beaker repository Integration for the native Toolathlon benchmark."""

from __future__ import annotations

import asyncio
import json
import os
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
from toolathlon_setup import (
    build_quickstart_row,
    collect_workspace_files,
    evaluate_workspace,
    prepare_toolathlon,
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

import toolathlon_baseline.runner as _runner

_orig_execute = _runner.execute


def _debug_execute(command, **kwargs):
    cmd = list(command)
    if "main.py" in cmd and "--debug" not in cmd:
        cmd.append("--debug")
    return _orig_execute(cmd, **kwargs)


_runner.execute = _debug_execute

request = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
model_params = pathlib.Path(request["model_params"]) if request.get("model_params") else None
summary = _runner.run_task(
    cache=pathlib.Path(request["cache"]),
    output=pathlib.Path(request["output"]),
    task=request["task"],
    model=request["model"],
    max_steps=int(request["max_steps"]),
    timeout=int(request["timeout"]),
    model_params=model_params,
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
        self.cache, self.source = prepare_toolathlon()
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


def _provider_name(model_name: str, base_url: str) -> str:
    if ":" in model_name:
        return model_name.split(":", 1)[0]
    if "/" in model_name:
        return model_name.split("/", 1)[0]
    if "openrouter.ai" in base_url:
        return "openrouter"
    if "anthropic.com" in base_url:
        return "anthropic"
    return "openai"


def _parse_tool_arguments(raw_args: Any) -> Any:
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            return {"raw": raw_args}
    return raw_args if raw_args is not None else {}


def _emit_trajectory_trace(
    *,
    runtime: RolloutRuntime[Any],
    trajectory: Mapping[str, Any],
    model_name: str,
    base_url: str,
) -> None:
    """Record model and tool execution spans reconstructed from the Toolathlon trajectory."""
    config = trajectory.get("config")
    system_prompts = config.get("system_prompts") if isinstance(config, Mapping) else None
    system_prompt = system_prompts.get("agent") if isinstance(system_prompts, Mapping) else None

    messages = trajectory.get("messages")
    if not isinstance(messages, list):
        return

    assistant_indices = [
        idx for idx, item in enumerate(messages) if isinstance(item, Mapping) and item.get("role") == "assistant"
    ]
    num_assistant = len(assistant_indices)
    if num_assistant == 0:
        return

    agent_cost = trajectory.get("agent_cost") if isinstance(trajectory.get("agent_cost"), Mapping) else {}
    key_stats = trajectory.get("key_stats") if isinstance(trajectory.get("key_stats"), Mapping) else {}
    total_in = int(agent_cost.get("total_input_tokens") or key_stats.get("input_tokens") or 0)
    total_out = int(agent_cost.get("total_output_tokens") or key_stats.get("output_tokens") or 0)
    raw_cached = agent_cost.get("total_cached_input_tokens")
    total_cached = int(raw_cached) if isinstance(raw_cached, int) and raw_cached >= 0 else None

    provider = _provider_name(model_name, base_url)
    context_messages: list[dict[str, Any]] = []
    if isinstance(system_prompt, str) and system_prompt.strip():
        context_messages.append({"role": "system", "content": system_prompt})

    tool_calls_by_id: dict[str, tuple[str, Any]] = {}
    assistant_Ordinal = 0

    for item in messages:
        if not isinstance(item, Mapping):
            continue
        role = item.get("role")
        if role == "assistant":
            in_tokens = (total_in // num_assistant) + (1 if assistant_Ordinal < (total_in % num_assistant) else 0)
            out_tokens = (total_out // num_assistant) + (1 if assistant_Ordinal < (total_out % num_assistant) else 0)
            cached_tokens = (
                (total_cached // num_assistant) + (1 if assistant_Ordinal < (total_cached % num_assistant) else 0)
                if total_cached is not None
                else None
            )
            assistant_Ordinal += 1

            with runtime.trace.model_call(
                operation="chat.completions",
                provider=provider,
                model=model_name,
                input_messages=list(context_messages),
            ) as call:
                call.output(dict(item))
                call.usage(
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    cached_input_tokens=cached_tokens,
                )

            raw_tool_calls = item.get("tool_calls")
            if isinstance(raw_tool_calls, list):
                for tc in raw_tool_calls:
                    if not isinstance(tc, Mapping):
                        continue
                    call_id = str(tc.get("id") or "")
                    fn = tc.get("function") if isinstance(tc.get("function"), Mapping) else {}
                    fn_name = str(fn.get("name") or "tool")
                    fn_args = _parse_tool_arguments(fn.get("arguments"))
                    if call_id:
                        tool_calls_by_id[call_id] = (fn_name, fn_args)
            context_messages.append(dict(item))
        elif role == "tool":
            call_id = str(item.get("tool_call_id") or "")
            tool_name, tool_args = tool_calls_by_id.get(call_id, ("tool", {}))
            with runtime.trace.tool_call(
                tool_name,
                arguments=tool_args,
                call_id=call_id or None,
            ) as tool_op:
                tool_op.output(item.get("content"))
            context_messages.append(dict(item))
        else:
            context_messages.append(dict(item))


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime[Any]) -> CaseResult:
    payload = TaskInput.model_validate(case_input)
    task = payload.task
    cache, _ = prepare_toolathlon()
    base_url, api_key, model_name, is_gateway = _resolve_model_target(runtime)

    with runtime.trace.stage(
        "toolathlon.run_task",
        inputs={"task": task, "model": model_name, "max_steps": DEFAULT_MAX_STEPS},
    ) as stage:
        with tempfile.TemporaryDirectory(prefix=f"beaker-toolathlon-case-{task}-") as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "run"
            request_path = root / "request.json"
            response_path = root / "response.json"

            request_payload: dict[str, Any] = {
                "cache": str(cache),
                "output": str(output_dir),
                "task": task,
                "model": model_name,
                "max_steps": DEFAULT_MAX_STEPS,
                "timeout": DEFAULT_TIMEOUT_SECONDS,
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
            src_dir = str(Path(__file__).resolve().parents[1] / "src")
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
            )
            stdout_bytes, stderr_bytes = await process.communicate()
            if process.returncode != 0 or not response_path.is_file():
                detail = (stderr_bytes or stdout_bytes).decode("utf-8", errors="replace")[-4000:]
                raise RetryableCaseError(f"Toolathlon runner subprocess failed (exit {process.returncode}): {detail}")

            summary = json.loads(response_path.read_text(encoding="utf-8"))
            traj_path = output_dir / "artifacts/traj_log.json"
            trajectory: dict[str, Any] = (
                json.loads(traj_path.read_text(encoding="utf-8")) if traj_path.is_file() else {}
            )

            _emit_trajectory_trace(
                runtime=runtime,
                trajectory=trajectory,
                model_name=model_name,
                base_url=base_url,
            )

            status = summary.get("status")
            agent_status = summary.get("agent_status") or trajectory.get("status")
            key_stats = summary.get("key_stats") or trajectory.get("key_stats") or {}
            llm_requests = int(key_stats.get("agent_llm_requests") or 0) if isinstance(key_stats, Mapping) else 0

            if status == "timeout" or agent_status == "timeout":
                stage.output({"status": "timeout", "llm_requests": llm_requests})
                raise RetryableCaseError(f"Toolathlon task {task!r} timed out after {DEFAULT_TIMEOUT_SECONDS}s")

            if llm_requests == 0 and agent_status != "success":
                log_path = output_dir / "run.log"
                log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:] if log_path.is_file() else ""
                stage.output({"status": "error_before_first_turn", "log_tail": log_tail})
                raise RetryableCaseError(f"Toolathlon task {task!r} exited before completing a model turn: {log_tail}")

            workspace_dir = output_dir / "artifacts/workspace"
            workspace_files = collect_workspace_files(workspace_dir)

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
    agent_status = str(context.get("agent_status") or "") or None
    raw_files = context.get("workspace_files")
    workspace_files = (
        {str(k): str(v) for k, v in raw_files.items() if isinstance(k, str) and isinstance(v, str)}
        if isinstance(raw_files, Mapping)
        else {}
    )

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
    targets=repository(("src/toolathlon_baseline",)),
    run_setup=Setup,
    run_case=run_case,
    score_case=score_case,
)
