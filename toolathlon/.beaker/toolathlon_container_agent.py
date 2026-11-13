"""Trace the original Toolathlon prepared-agent phase, without tracing its grader."""

from __future__ import annotations

import asyncio

from scripts.containerized.container_agent import consume_task_bundle, parse_args, run_agent_phase
from toolathlon_native_trace import export_native_trace
from utils.general.helper import setup_proxy
from utils.roles.task_agent import TaskStatus


async def main() -> int:
    args = parse_args()
    setup_proxy(args.with_proxy)
    bundle = consume_task_bundle(args.bundle_file)
    async with export_native_trace():
        status = await run_agent_phase(bundle, debug=args.debug, allow_resume=args.allow_resume, manual=args.manual)
    return 0 if status == TaskStatus.SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
