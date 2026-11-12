"""Toolathlon's native OpenAI Agents 0.20 entrypoint, using public SDK APIs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import traceback
from contextlib import AsyncExitStack
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import yaml
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, RunConfig, Runner
from agents.exceptions import MaxTurnsExceeded
from agents.mcp import MCPServerStdio
from openai import AsyncOpenAI, BadRequestError
from openai.types.shared import Reasoning
from toolathlon_context import Conversation
from toolathlon_native_trace import export_native_trace
from toolathlon_tools import LOCAL_TOOLS, local_tools, mcp_tool
from utils.openai_agents_monkey_patch.tool_name_aliases import (
    build_tool_name_aliases,
    rewrite_tool_name_references,
    validate_model_tool_names,
)


def model_settings(config: dict) -> ModelSettings:
    params_file = os.environ.get("TOOLATHLON_MODEL_PARAMS_FILE")
    if params_file:
        params = json.loads(Path(params_file).read_text())
    else:
        params = {
            "temperature": 0.6,
            "top_p": 1.0,
            **config["agent"]["generation"],
            **{key: config["agent"]["tool"][key] for key in ("tool_choice", "parallel_tool_calls")},
        }
    if not isinstance(params, dict):
        raise ValueError("Model parameters must be a JSON object")
    if any(key in params for key in ("model", "messages", "tools", "stream")):
        raise ValueError("Model parameters cannot override the agent's model, messages, tools or stream mode")
    if "reasoning_effort" in params:
        params["reasoning"] = Reasoning(effort=params.pop("reasoning_effort"))
    if isinstance(params.get("reasoning"), dict):
        params["reasoning"] = Reasoning(**params["reasoning"])
    known = {field.name for field in fields(ModelSettings)}
    settings = {key: value for key, value in params.items() if key in known}
    extra = {key: value for key, value in params.items() if key not in known}
    if extra:
        settings["extra_args"] = {**(settings.get("extra_args") or {}), **extra}
    return ModelSettings(**settings)


async def run(args) -> None:
    checkout = Path.cwd()
    config = json.loads(Path(args.eval_config).read_text())
    task = checkout / "tasks" / args.task_dir
    requirements = json.loads((task / "task_config.json").read_text())
    artifacts = Path(config["global_task_config"]["dump_path"])
    artifacts.mkdir(parents=True, exist_ok=True)
    workspace = artifacts / "workspace"
    if (task / "initial_workspace").is_dir():
        shutil.copytree(task / "initial_workspace", workspace)
    else:
        workspace.mkdir()
    (workspace / "arxiv_local_storage").mkdir(exist_ok=True)
    launch_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if (preprocess := task / "preprocess/main.py").is_file():
        subprocess.run(
            [sys.executable, str(preprocess), "--agent_workspace", str(workspace), "--launch_time", launch_time],
            check=True,
        )
    instruction = (task / "docs/task.md").read_text().strip()
    instructions = (task / "docs/agent_system_prompt.md").read_text()
    for placeholder, value in {
        "current_working_dir": str(checkout),
        "workspace_dir": str(workspace),
        "workspace_dir_rela": os.path.relpath(workspace, checkout),
        "time": launch_time,
    }.items():
        instructions = instructions.replace(f"!!<<<<||||{placeholder}||||>>>>!!", value)
    aliases = build_tool_name_aliases(tool for group in LOCAL_TOOLS.values() for tool in group)
    aliases["local-claim-done"] = "local_claim_done"
    instructions = rewrite_tool_name_references(instructions, aliases)
    instructions += (
        "\nPlease complete the given task independently. "
        "Do not seek confirmation or additional feedback from the user. "
        "You should handle all situations on your own, as the user will not provide any further information."
    )
    context_limit = config["agent"]["model"].get("context_window") or 128000
    conversation = Conversation(workspace, instruction, context_limit)
    status = "failed"
    try:
        async with export_native_trace(), AsyncExitStack() as servers:
            tools = local_tools(requirements.get("needed_local_tools", []))
            for name in requirements["needed_mcp_servers"]:
                paths = list((checkout / "configs/mcp_servers").glob("*.yaml"))
                definition = next(
                    yaml.safe_load(path.read_text())
                    for path in paths
                    if yaml.safe_load(path.read_text()).get("name", path.stem) == name
                )
                params = json.loads(json.dumps(definition["params"]).replace("${agent_workspace}", str(workspace)))
                server = await servers.enter_async_context(
                    MCPServerStdio(
                        name=name,
                        params=params,
                        cache_tools_list=definition.get("cache_tools_list", True),
                        client_session_timeout_seconds=definition.get("client_session_timeout_seconds", 100),
                    )
                )
                tools.extend(mcp_tool(server, tool) for tool in await server.list_tools())
            validate_model_tool_names(tools)
            async with AsyncOpenAI(
                base_url=os.environ["TOOLATHLON_OPENAI_BASE_URL"], api_key=os.environ["TOOLATHLON_OPENAI_API_KEY"]
            ) as client:
                agent = Agent(
                    name="Assistant",
                    instructions=instructions,
                    tools=tools,
                    model=OpenAIChatCompletionsModel(model=args.model_short_name, openai_client=client),
                    model_settings=model_settings(config),
                    tool_use_behavior={"stop_at_tool_names": ["local_claim_done"]},
                )
                run_config = RunConfig(
                    tracing_disabled=False,
                    trace_include_sensitive_data=True,
                    call_model_input_filter=conversation.filter_input,
                )
                inputs = [{"role": "user", "content": instruction}]
                os.chdir(workspace)
                while conversation.attempts < args.max_steps_under_single_turn_mode:
                    try:
                        await Runner.run(
                            agent,
                            inputs,
                            context=conversation.context,
                            hooks=conversation,
                            run_config=run_config,
                            max_turns=args.max_steps_under_single_turn_mode - conversation.attempts,
                        )
                        status = "success"
                        break
                    except BadRequestError as exc:
                        if not any(
                            marker in str(exc).lower()
                            for marker in (
                                "context_length_exceeded",
                                "maximum context length",
                                "context window",
                                "messages_too_long",
                            )
                        ):
                            raise
                        inputs = conversation.reset(instruction)
                else:
                    status = "max_turns_reached"
    except MaxTurnsExceeded:
        status = "max_turns_reached"
    except asyncio.CancelledError:
        status = "interrupted"
        raise
    except Exception:
        traceback.print_exc()
    finally:
        os.chdir(checkout)
        stats = dict(conversation.stats)
        stats.update(
            total_tokens=stats["input_tokens"] + stats["output_tokens"],
            non_cached_input_tokens=stats["input_tokens"] - stats["cached_input_tokens"],
            truncations=len(conversation.meta["truncation_history"]),
        )
        trajectory = {
            "status": status,
            "messages": conversation.messages,
            "key_stats": stats,
            "config": {"task_dir": args.task_dir, "system_prompts": {"agent": instructions}},
            "agent_framework": "openai-agents==0.20.0",
        }
        (artifacts / "traj_log.json").write_text(json.dumps(trajectory, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_config", required=True)
    parser.add_argument("--task_dir", required=True)
    parser.add_argument("--model_short_name", required=True)
    parser.add_argument("--max_steps_under_single_turn_mode", required=True, type=int)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
