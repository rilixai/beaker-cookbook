"""Prepare and run the official Toolathlon baseline without Beaker."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from toolathlon_baseline import QUICKSTART_TASKS, UPSTREAM_REVISION
from toolathlon_baseline.prepare import prepare
from toolathlon_baseline.runner import check_tools, read_result, run_task


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Fetch pinned Toolathlon and install its Python/MCP dependencies")
    prep.add_argument("--cache", type=Path, default=Path(".cache/toolathlon"))
    check = commands.add_parser("check", help="Connect to actual MCP servers without making model calls")
    check.add_argument("--cache", type=Path, default=Path(".cache/toolathlon"))
    check.add_argument("--task", choices=QUICKSTART_TASKS, default="find-alita-paper")
    check.add_argument("--output", type=Path, required=True, help="A new directory for tool listings and setup logs")
    run = commands.add_parser(
        "run", help="Execute the original agent loop and official evaluator (incurs API charges)"
    )
    run.add_argument("--cache", type=Path, default=Path(".cache/toolathlon"))
    run.add_argument("--task", choices=QUICKSTART_TASKS, default="find-alita-paper")
    run.add_argument("--model", required=True, help="Exact model ID accepted by your OpenAI-compatible endpoint")
    run.add_argument(
        "--output", type=Path, required=True, help="A new directory for source, workspace, logs, and verdict"
    )
    run.add_argument("--max-steps", type=positive_int, default=100)
    run.add_argument("--timeout", type=positive_int, default=1800, help="Total execution budget in seconds")
    run.add_argument(
        "--system-prompt", type=Path, help="Optional replacement for the task's original agent system prompt"
    )
    run.add_argument("--model-params", type=Path, help="Optional JSON object of provider request parameters")
    results = commands.add_parser("results", help="Read a saved verdict without making model calls")
    results.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            source = prepare(args.cache)
            print(json.dumps({"source": str(source), "upstream_revision": UPSTREAM_REVISION}, indent=2))
            return
        if args.command == "check":
            print(json.dumps(check_tools(cache=args.cache, output=args.output, task=args.task), indent=2))
            return
        if args.command == "results":
            summary = read_result(args.output)
        else:
            summary = run_task(
                cache=args.cache,
                output=args.output,
                task=args.task,
                model=args.model,
                max_steps=args.max_steps,
                timeout=args.timeout,
                system_prompt=args.system_prompt,
                model_params=args.model_params,
            )
        print(json.dumps(summary, indent=2))
        raise SystemExit({"passed": 0, "failed": 1}.get(summary["status"], 2))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"toolathlon-baseline: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
