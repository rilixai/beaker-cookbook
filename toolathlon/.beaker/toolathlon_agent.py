"""The pinned upstream single-turn launch path, without in-agent evaluation."""

import argparse
import asyncio
import json

from utils.data_structures.task_config import TaskConfig
from utils.openai_agents_monkey_patch.custom_mcp_util import *  # noqa: F403
from utils.openai_agents_monkey_patch.custom_run_impl import *  # noqa: F403
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
