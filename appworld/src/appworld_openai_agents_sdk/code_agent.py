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
from functools import wraps
from typing import Any, cast

from agents import (
    Agent,
    AgentsException,
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
from appworld.common.random import set_random_seed
from appworld.task import Task
from jinja2 import Template
from openai import OpenAIError
from rich import print

from appworld_openai_agents_sdk.models import ModelProfile
from appworld_openai_agents_sdk.vendored.common.logger import Logger


set_tracing_disabled(True)


# Scoped to one task-runner invocation; values are snapshots of successful
# documentation responses, never credentials, account records or REPL state.
DocumentationCache = dict[tuple[str, bool, str, str], str]


def reuse_api_documentation(world: AppWorld, cache: DocumentationCache) -> str:
    """Supply previously discovered interfaces and observe fresh doc requests.

    AppWorld's API interfaces are fixed within this process. Restrict reuse to
    the current data version, schema setting and available apps. Explicit API
    calls still execute normally, including errors; this only observes them.
    """
    if world.remote_environment_url:
        return ""  # The remote interpreter owns a separate API collection.
    version = world.task.db_version
    schemas = world.show_api_response_schemas
    allowed_apps = set(world.apis.keys())
    available: dict[str, dict[str, Any]] = {}
    for (doc_version, doc_schemas, app, name), value in cache.items():
        if (doc_version, doc_schemas) == (version, schemas) and app in allowed_apps:
            app_docs = available.setdefault(app, {"specifications": {}})
            if name:
                app_docs["specifications"][name] = json.loads(value)
            else:
                app_docs["api_descriptions"] = json.loads(value)

    def observe(api_name: str) -> None:
        original = world.apis.api_docs[api_name]

        @wraps(original)
        def observed(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            app = kwargs.get("app_name")
            name = kwargs.get("api_name", "") if api_name == "show_api_doc" else ""
            is_list = (
                api_name == "show_api_descriptions"
                and isinstance(result, list)
                and bool(result)
                and all(isinstance(item, dict) and {"name", "description"} <= item.keys() for item in result)
            )
            is_spec = (
                api_name == "show_api_doc"
                and isinstance(result, dict)
                and {"app_name", "api_name", "path", "method", "description", "parameters"} <= result.keys()
                and (not schemas or "response_schemas" in result)
                and result["app_name"] == app
                and result["api_name"] == name
            )
            if isinstance(app, str) and app in allowed_apps and isinstance(name, str) and (is_list or is_spec):
                try:
                    snapshot = json.dumps(result, separators=(",", ":"))
                except (TypeError, ValueError):
                    pass  # Recording must not change the API's return behavior.
                else:
                    key = (version, schemas, app, name)
                    cache.pop(key, None)
                    cache[key] = snapshot
                    # Bound accumulated context for long, heterogeneous runs.
                    # Evicted documentation remains discoverable normally.
                    while sum(map(len, cache.values())) > 64_000:
                        cache.pop(next(iter(cache)))
            return result

        world.apis.api_docs[api_name] = observed

    observe("show_api_descriptions")
    observe("show_api_doc")
    if not available:
        return ""
    return (
        "\n\nPreviously retrieved API documentation for the available apps follows. "
        "Read these complete interfaces instead of requesting the same documentation again; "
        "discover any missing APIs or specifications normally. Only documentation is reused. "
        "Retrieve this task's credentials and account data afresh; no Python variables, "
        "tokens, people, IDs or task results carry over.\n"
        + json.dumps(available, separators=(",", ":"))
    )


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
        # Keep namespaced OpenRouter IDs intact through the SDK's provider router.
        model=f"openai/{profile.name}" if profile.api_type == "chat_completions" else profile.name,
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
    *,
    run_config: RunConfig | None = None,
    raise_provider_errors: bool = False,
    documentation_cache: DocumentationCache | None = None,
) -> None:
    with AppWorld(task_id=task_id) as world:
        logger.start_task(world)
        instructions = render_instructions(prompt_file_path, world)
        if documentation_cache is not None:
            instructions += reuse_api_documentation(world, documentation_cache)
        agent, step_counter = build_agent(world, profile, instructions, logger)
        run_config = run_config or RunConfig(tracing_disabled=True)
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
            # A model/API failure (malformed tool call, provider error) ends
            # this task as a failure but must not abort the experiment.
            except (AgentsException, OpenAIError) as error:
                if raise_provider_errors and isinstance(error, OpenAIError):
                    raise
                print(f"Task {task_id} aborted: {error!r}")
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
    *,
    run_config: RunConfig | None = None,
    raise_provider_errors: bool = False,
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
        documentation_cache: DocumentationCache = {}
        for task_id in task_ids:
            # Upstream reseeds per task (Agent.initialize); mirror it.
            set_random_seed(appworld_config.get("random_seed"))
            await run_code_agent_on_task(
                task_id=task_id,
                profile=profile,
                logger=logger,
                prompt_file_path=prompt_file_path,
                max_steps=max_steps,
                run_config=run_config,
                raise_provider_errors=raise_provider_errors,
                documentation_cache=documentation_cache,
            )
