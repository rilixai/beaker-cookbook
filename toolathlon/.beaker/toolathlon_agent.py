"""Toolathlon single-turn launch with OpenAI Agents 0.20 and native OTLP tracing."""

import argparse
import asyncio
import json

from toolathlon_native_trace import export_native_trace
from toolathlon_sdk_compat import install
from utils.data_structures.task_config import TaskConfig
from utils.openai_agents_monkey_patch.custom_mcp_util import *  # noqa: F403
from utils.task_runner.runner import TaskRunner


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_config", required=True)
    parser.add_argument("--task_dir", required=True)
    parser.add_argument("--model_short_name", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--max_steps_under_single_turn_mode", required=True, type=int)
    args = parser.parse_args()
    with open(args.eval_config) as stream:
        config = json.load(stream)
    config["agent"]["model"] = {"short_name": args.model_short_name, "provider": args.provider}
    config["global_task_config"]["max_steps_under_single_turn_mode"] = args.max_steps_under_single_turn_mode
    mcp, agent, user = TaskRunner.load_configs(config)
    task = TaskConfig.build(args.task_dir, agent.model.short_name, config["global_task_config"], True, False)
    install()
    async with export_native_trace():
        await TaskRunner.run_single_task(
            task_config=task,
            agent_config=agent,
            user_config=user,
            mcp_config=mcp,
            debug=True,
            single_turn_mode=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
