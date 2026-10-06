"""One AppWorld case per interpreter; invoked only by the Beaker integration."""

from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from appworld_setup import PROJECT, prepare_appworld
from beaker import CaseResult, RetryableCaseError
from beaker.tracing.core import LocalCaptureManager
from beaker.tracing.integrations import openai_agents


class _EventLoop(asyncio.SelectorEventLoop):
    def time(self) -> float:
        # AppWorld freezes Python clocks. Network timers must use real elapsed time.
        return time.clock_gettime(time.CLOCK_MONOTONIC)


async def run_in_process(*, case_input, runtime) -> CaseResult:
    prepare_appworld()
    from agents import set_trace_processors, set_tracing_disabled
    from agents.run import RunConfig
    from agents.tracing import get_trace_provider
    from appworld import AppWorld, evaluate_task
    from appworld.apps.lib.models.db import CachedDBHandler

    from appworld_openai_agents_sdk.code_agent import run_code_agent_on_tasks
    from appworld_openai_agents_sdk.models import ModelProfile
    from appworld_openai_agents_sdk.runner import MAX_STEPS, PROMPTS_DIR, RANDOM_SEED

    profile = ModelProfile.from_toml(PROJECT / "configs/model.toml")
    if runtime.model:
        provider, name = runtime.model.split(":", 1)
        if provider != "openai":
            raise ValueError("This application supports OpenAI models only")
        profile = ModelProfile(name=name)
    task_ids = [task["task_id"] for task in case_input["tasks"]]
    experiment = f"beaker-{uuid4().hex}"
    # The application disables SDK telemetry by default. Enable it only in this
    # isolated evaluation, and keep traces in Beaker rather than OpenAI export.
    provider = get_trace_provider()
    processors = list(provider._multi_processor._processors)
    disabled = provider._disabled
    try:
        set_trace_processors([])
        set_tracing_disabled(False)
        with openai_agents.registered(runtime.trace):
            await run_code_agent_on_tasks(
                experiment_name=experiment,
                task_ids=task_ids,
                profile=profile,
                prompt_file_path=str(PROMPTS_DIR / "react_code_agent/instructions.txt"),
                appworld_config={"random_seed": RANDOM_SEED},
                logger_config={"color": False, "verbose": False},
                max_steps=MAX_STEPS,
                run_config=RunConfig(tracing_disabled=False),
                raise_provider_errors=True,
            )
    finally:
        set_trace_processors(processors)
        set_tracing_disabled(disabled)
        AppWorld.close_all()
    results = {}
    for task_id in task_ids:
        try:
            results[task_id] = evaluate_task(task_id, experiment, save_report=False).to_dict()
        finally:
            CachedDBHandler.reset()
    return CaseResult(output={"tasks": results}, output_kind="record")


def main() -> None:
    sys.path.insert(0, str(PROJECT / "src"))
    directory = Path(sys.argv[1])
    request = json.loads((directory / "request.json").read_text())
    manager = LocalCaptureManager(directory / "traces")
    session = manager.capture(case_id=request["identity"].get("beaker.case.id"))
    session.capture_id = request["identity"].get("beaker.capture.id", session.capture_id)
    session.candidate_id = request["identity"].get("beaker.candidate.id")
    run_id = request["identity"].get("beaker.run.id")
    if run_id:
        manager.run_id = run_id
    runtime = SimpleNamespace(model=request["model"], trace=session.trace)
    try:
        # run_in_process scopes the Agents adapter around execution, excluding scoring.
        with session, asyncio.Runner(loop_factory=_EventLoop) as runner:
            result = runner.run(run_in_process(case_input=request["input"], runtime=runtime))
        response = {"output": result.output, "output_kind": result.output_kind}
    except Exception as exc:
        response = {
            "error": f"{type(exc).__name__}: {exc}",
            "retryable": isinstance(exc, RetryableCaseError),
            "traceback": traceback.format_exc(),
        }
    artifacts = [ref.to_dict() for ref in session.artifacts]
    for ref in artifacts:
        if not ref.get("inline"):
            ref.pop("content", None)
    (directory / "artifacts.json").write_text(json.dumps(artifacts))
    (directory / "response.json").write_text(json.dumps(response))


if __name__ == "__main__":
    main()
