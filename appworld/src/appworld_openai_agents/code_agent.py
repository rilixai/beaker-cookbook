"""ReAct-style code-execution agent on the OpenAI Agents SDK.

The agent gets a single ``execute_python`` tool backed by AppWorld's own
in-process interpreter (``world.execute``), where the apps are callable as
``apis.<app>.<api>(...)``. Nothing is pre-selected for it: the agent discovers
the APIs it needs at runtime through the ``api_docs`` app, exactly like the
AppWorld paper's ReAct baselines. The instructions (with the worked demo task)
are adapted from upstream's ``react_code_agent`` prompt — see ATTRIBUTION.md.

An episode ends when the agent runs ``apis.supervisor.complete_task(...)`` in
code (checked after every execution) or exhausts its step budget.
"""

from __future__ import annotations

import json
from typing import Any, cast

from agents import (
    Agent,
    FunctionToolResult,
    MaxTurnsExceeded,
    RunContextWrapper,
    Runner,
    function_tool,
    set_default_openai_api,
    set_tracing_disabled,
)
from agents.agent import ToolsToFinalOutputResult
from agents.model_settings import ModelSettings
from agents.run import RunConfig
from appworld import AppWorld
from appworld.common.io import read_file
from appworld.task import Task
from jinja2 import Template
from rich import print

from appworld_openai_agents.models import ModelProfile
from appworld_openai_agents.vendored.common.logger import Logger


set_tracing_disabled(True)


def render_instructions(prompt_file_path: str, world: AppWorld) -> str:
    template = Template(cast(str, read_file(prompt_file_path)))
    app_descriptions = json.dumps(
        [{"name": k, "description": v} for (k, v) in world.task.app_descriptions.items()],
        indent=1,
    )
    return cast(
        str,
        template.render(
            instruction=world.task.instruction,
            main_user=world.task.supervisor,
            app_descriptions=app_descriptions,
        ),
    )


def build_agent(
    world: AppWorld, profile: ModelProfile, instructions: str, logger: Logger
) -> tuple[Agent, dict[str, int]]:
    step_counter = {"count": 0}

    # NOTE: async so the SDK runs it on the event-loop thread. AppWorld's
    # interpreter must run on the thread that created the world (its execution
    # hangs from a worker thread, which is where the SDK runs sync tools).
    @function_tool
    async def execute_python(code: str) -> str:
        """Execute python code in the task's REPL environment and return its
        printed output. Variables persist across calls. Apps are callable as
        apis.<app_name>.<api_name>(...)."""
        step_counter["count"] += 1
        logger.show_message(role="agent", content=code, step_number=step_counter["count"])
        output = cast(str, world.execute(code))
        logger.show_message(role="environment", content=output, step_number=step_counter["count"])
        return output

    def stop_when_task_completed(
        context: RunContextWrapper[Any], tool_results: list[FunctionToolResult]
    ) -> ToolsToFinalOutputResult:
        if world.task_completed():
            return ToolsToFinalOutputResult(is_final_output=True, final_output=str(tool_results[-1].output))
        return ToolsToFinalOutputResult(is_final_output=False)

    settings = profile.settings()
    # One code chunk per step (the prompt's contract); parallel calls to a
    # single REPL would interleave nondeterministically.
    settings["parallel_tool_calls"] = False
    agent = Agent(
        name="Assistant",
        model=profile.name,
        model_settings=ModelSettings(**settings),
        instructions=instructions,
        tools=[execute_python],
        tool_use_behavior=stop_when_task_completed,
        reset_tool_choice=False,
    )
    return agent, step_counter


async def run_code_agent_on_task(
    task_id: str,
    profile: ModelProfile,
    logger: Logger,
    prompt_file_path: str,
    max_steps: int,
) -> None:
    with AppWorld(task_id=task_id) as world:
        logger.start_task(world)
        instructions = render_instructions(prompt_file_path, world)
        agent, step_counter = build_agent(world, profile, instructions, logger)
        run_config = RunConfig(tracing_disabled=True)
        input_: Any = "Begin. Submit your first code step with the execute_python tool."
        while not world.task_completed() and step_counter["count"] < max_steps:
            try:
                result = await Runner.run(
                    starting_agent=agent,
                    input=input_,
                    max_turns=max_steps - step_counter["count"],
                    run_config=run_config,
                )
            except MaxTurnsExceeded:
                world.save_state()
                break
            world.save_state()
            if world.task_completed():
                break
            # The model produced a plain message instead of a tool call; nudge
            # it to keep going within the remaining budget (counted as a step
            # so a chatty model cannot loop forever).
            step_counter["count"] += 1
            input_ = result.to_input_list() + [
                {
                    "role": "user",
                    "content": (
                        "Continue with the task. Submit code via the execute_python tool; "
                        "call apis.supervisor.complete_task(...) when done."
                    ),
                }
            ]
        logger.complete_task()


async def run_code_agent_on_tasks(
    experiment_name: str,
    task_ids: list[str],
    profile: ModelProfile,
    prompt_file_path: str,
    appworld_config: dict[str, Any],
    logger_config: dict[str, Any],
    max_steps: int,
) -> None:
    print(f"Running Experiment: {experiment_name}")
    set_default_openai_api(profile.api_type)
    print("Loading test tasks...")
    for task_id in task_ids:
        Task.load(task_id=task_id)
    logger = Logger(**logger_config)
    logger.initialize(
        experiment_name=experiment_name,
        num_tasks=len(task_ids),
        num_processes=1,
        process_index=0,
    )
    with AppWorld.initializer(update_defaults=True, experiment_name=experiment_name, **appworld_config):
        for task_id in task_ids:
            await run_code_agent_on_task(
                task_id=task_id,
                profile=profile,
                logger=logger,
                prompt_file_path=prompt_file_path,
                max_steps=max_steps,
            )
