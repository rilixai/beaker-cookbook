"""Prepare and run the official Toolathlon baseline without Beaker."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from toolathlon_baseline import QUICKSTART_TASKS, UPSTREAM_REVISION
from toolathlon_baseline.catalog import task_catalog
from toolathlon_baseline.containerized import DEFAULT_IMAGE, check_environment, configured_source, run_containerized
from toolathlon_baseline.prepare import prepare, prepared_source
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
    catalog = commands.add_parser("tasks", help="List every official task and its tool requirements")
    catalog.add_argument("--cache", type=Path, default=Path(".cache/toolathlon"))
    catalog.add_argument("--environment", type=Path, help="Alternatively read a configured upstream checkout")
    full_check = commands.add_parser("check-env", help="Check the configured Linux host, runtime, and image")
    full_check.add_argument("--environment", type=Path, required=True)
    full_check.add_argument("--image", default=DEFAULT_IMAGE)
    check = commands.add_parser("check", help="Connect to actual MCP servers without making model calls")
    check.add_argument("--cache", type=Path, default=Path(".cache/toolathlon"))
    check.add_argument("--task", choices=QUICKSTART_TASKS, default="find-alita-paper")
    check.add_argument("--output", type=Path, required=True, help="A new directory for tool listings and setup logs")
    run = commands.add_parser(
        "run", help="Execute the original agent loop and official evaluator (incurs API charges)"
    )
    run.add_argument("--cache", type=Path, default=Path(".cache/toolathlon"))
    selection = run.add_mutually_exclusive_group()
    selection.add_argument("--task", action="append", help="Official task ID; repeat to select several tasks")
    selection.add_argument("--all", action="store_true", help="Run all official tasks (108 at the pinned revision)")
    run.add_argument("--runner", choices=("auto", "native", "containerized"), default="auto")
    run.add_argument("--environment", type=Path, help="Configured upstream checkout for the full containerized runner")
    run.add_argument("--workers", type=positive_int, default=1, help="Concurrent containerized tasks (default: 1)")
    run.add_argument("--image", default=DEFAULT_IMAGE, help="Container image for the full runner")
    run.add_argument("--model", required=True, help="Exact model ID accepted by your OpenAI-compatible endpoint")
    run.add_argument(
        "--output", type=Path, required=True, help="A new directory for source, workspace, logs, and verdict"
    )
    run.add_argument("--max-steps", type=positive_int, default=100)
    run.add_argument("--timeout", type=positive_int, help="Per-task seconds (native: 1800; containerized: 5400)")
    run.add_argument(
        "--system-prompt", type=Path, help="Optional system prompt replacement for a native quickstart run"
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
        if args.command == "tasks":
            source = configured_source(args.environment) if args.environment else prepared_source(args.cache)
            print(json.dumps(task_catalog(source), indent=2))
            return
        if args.command == "check-env":
            print(json.dumps(check_environment(args.environment, args.image), indent=2))
            return
        if args.command == "results":
            summary = read_result(args.output)
        else:
            selected = None if args.all else args.task or ["find-alita-paper"]
            native = args.runner == "native" or (
                args.runner == "auto"
                and not args.environment
                and selected is not None
                and len(selected) == 1
                and selected[0] in QUICKSTART_TASKS
            )
            if native:
                if selected is None or len(selected) != 1 or selected[0] not in QUICKSTART_TASKS:
                    raise ValueError(
                        "Native mode supports one quickstart task; use --environment for other selections"
                    )
                if args.workers != 1:
                    raise ValueError("--workers applies to containerized runs")
                summary = run_task(
                    cache=args.cache,
                    output=args.output,
                    task=selected[0],
                    model=args.model,
                    max_steps=args.max_steps,
                    timeout=args.timeout or 1800,
                    system_prompt=args.system_prompt,
                    model_params=args.model_params,
                )
            else:
                if args.environment is None:
                    raise ValueError("Pass --environment PATH to a configured Toolathlon Linux checkout")
                if args.system_prompt is not None:
                    raise ValueError("--system-prompt is native-only; full runs use the configured checkout's prompts")
                summary = run_containerized(
                    environment=args.environment,
                    output=args.output,
                    tasks=selected,
                    model=args.model,
                    max_steps=args.max_steps,
                    workers=args.workers,
                    timeout=args.timeout or 5400,
                    image=args.image,
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
