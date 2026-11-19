"""Run the editable Terminus agent against fixed Terminal-Bench Science tasks and tests."""

from __future__ import annotations

import asyncio
import math
import tarfile
import tempfile
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from beaker import (
    Case,
    CaseResult,
    CaseScore,
    Check,
    Integration,
    JsonValue,
    RetryableCaseError,
    RolloutRuntime,
    RunSetup,
    SetupRuntime,
    inference_target,
    repository,
)
from beaker.tracing.integrations.litellm import registered
from benchmark_data import Row, TaskInput, stage_task
from docker_runtime import docker_runtime
from harbor.models.environment_type import EnvironmentType
from harbor.models.model_api_format import ModelApiFormat
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial


class Setup(RunSetup[Row]):
    row_model = Row

    async def load_cases(self, row: Row, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        del runtime
        yield Case(id=row.id, input=row.input.model_dump(), expected=row.expected, metadata=row.metadata)


@dataclass
class CandidateRuntime:
    scratch_dir: Path
    resources: AsyncExitStack
    startup_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    ready: bool = False

    async def start(self) -> None:
        # Only startup is serialized; all admitted task cases can overlap.
        async with self.startup_lock:
            if not self.ready:
                await self.resources.enter_async_context(docker_runtime(self.scratch_dir))
                self.ready = True


@asynccontextmanager
async def open_candidate(*, targets_dir: Path | None, scratch_dir: Path) -> AsyncIterator[CandidateRuntime]:
    del targets_dir
    async with AsyncExitStack() as resources:
        # Smoke loads data and opens this context without executing a case.
        # Start Docker only for execution, then close it when the rollout ends.
        yield CandidateRuntime(scratch_dir, resources)


def agent_config(runtime: RolloutRuntime[Any]) -> AgentConfig:
    values: dict[str, Any] = {
        "import_path": "terminal_bench_science.agent:CookbookTerminus",
        "model_name": "openai/gpt-6-luna",
        "model_api_format": ModelApiFormat.OPENAI_CHAT_COMPLETIONS,
    }
    if runtime.model:
        target = inference_target(runtime)
        # The gateway accepts Beaker's provider:model identifiers. Avoid
        # Harbor reinterpreting them as native provider model names.
        values.update(
            model_name=target.model,
            base_url=target.base_url,
            skip_model_parsing=True,
            model_api_key_name="OPENAI_API_KEY",
            env={"OPENAI_API_KEY": target.api_key},
        )
    return AgentConfig(**values)


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime[CandidateRuntime]) -> CaseResult:
    ref = TaskInput.model_validate(case_input)
    if runtime.candidate is None:
        raise RuntimeError("Candidate Docker runtime was not opened")
    await runtime.candidate.start()
    with tempfile.TemporaryDirectory(prefix="trial-", dir=runtime.candidate.scratch_dir) as temporary:
        work = Path(temporary)
        task_dir = work / "task"
        task_dir.mkdir()
        archive = await asyncio.to_thread(stage_task, ref.task_id, work)
        with tarfile.open(archive) as bundle:
            bundle.extractall(task_dir, filter="data")
        trial = await Trial.create(
            TrialConfig(
                task=TaskConfig(path=task_dir),
                trial_name=f"{ref.task_id.rsplit('/', 1)[-1]}__{uuid4().hex[:12]}",
                trials_dir=work / "trials",
                agent=agent_config(runtime),
                environment=EnvironmentConfig(type=EnvironmentType.DOCKER, delete=True),
            )
        )
        if (task_dir / "instruction.md").read_text() != ref.instruction:
            raise ValueError("Dataset instruction differs from the pinned upstream task")
        start = asyncio.get_running_loop().time()
        with runtime.trace.stage("science.task", inputs={"task_id": ref.task_id}) as stage:
            async with registered(runtime.trace) as traces:
                result = await trial.run()
                await traces.flush()
            # Harbor returns infrastructure/provider errors in its result. Do
            # not turn an unavailable sandbox or failed model call into score 0.
            for error in result.recorded_exceptions():
                if error.exception_type != "AgentTimeoutError":
                    message = f"{error.exception_type}: {error.exception_message}"
                    if error.policy.should_retry:
                        raise RetryableCaseError(message)
                    raise RuntimeError(message)
            if not result.reward_valid or result.verifier_result is None:
                raise RuntimeError("Harbor did not return a valid verifier reward")
            rewards = result.verifier_result.rewards
            if rewards is None or set(rewards) != {"reward"}:
                raise ValueError(f"Unexpected Terminal-Bench Science reward fields: {rewards}")
            reward = float(rewards["reward"])
            if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
                raise ValueError(f"Invalid Terminal-Bench Science reward: {reward}")
            diagnostics = []
            for path in sorted(trial.paths.trial_dir.rglob("test-stdout.txt")):
                diagnostics.append(path.read_text(errors="replace")[-12000:])
            output: dict[str, JsonValue] = {
                "task_id": ref.task_id,
                "reward": reward,
                "verifier_output": "\n".join(diagnostics)[-12000:],
                "agent_timed_out": bool(result.exception_info),
            }
            stage.output(output)
        return CaseResult(
            output=output,
            output_kind="record",
            context={
                "beaker_rollout_duration_ms": (asyncio.get_running_loop().time() - start) * 1000,
            },
        )


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    del case_files_dir
    output = result.output
    if not isinstance(output, dict) or output.get("task_id") != case.id:
        raise ValueError("Terminal-Bench Science output does not match the case")
    reward = output.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not 0 <= reward <= 1:
        raise ValueError("Expected a Terminal-Bench Science reward in [0, 1]")
    return CaseScore(
        objective=float(reward),
        field_scores={"task_success": float(reward)},
        checks=(
            Check(
                name=f"Terminal-Bench Science tests: {case.id}",
                verdict="pass" if reward == 1 else "fail",
                expected=1,
                predicted=reward,
                message=str(output.get("verifier_output") or "Upstream verifier completed"),
            ),
        ),
    )


integration = Integration(
    targets=repository(("src/terminal_bench_science",)),
    run_setup=Setup,
    open_candidate=open_candidate,
    run_case=run_case,
    score_case=score_case,
)
