"""CLI entrypoint for the AppWorld OpenAI-Agents baseline.

Run as ``uv run appworld-openai-agents <command> ...`` (or ``python -m
appworld_openai_agents.cli <command> ...``) after installing this project
from its directory (``cd appworld_openai_agents && uv sync --group dev``).

Subcommands:
* ``run`` — start the AppWorld servers and run the vendored OpenAI Agents SDK
  agent (with its api_predictor pass) over a split, writing per-task
  predictions/state under ``<output-dir>/experiments/outputs/<experiment>``.
* ``evaluate`` — score a finished run with AppWorld's evaluator and print
  TGC/SGC.

One-time setup (see README): ``uv run appworld install`` then
``uv run appworld download data`` (from this directory, so data lands under
the default ``--output-dir .``).
"""

from __future__ import annotations

import argparse
import os
import sys

from .config import DEFAULT_MAX_STEPS, RunSpec, build_runner_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AppWorld baseline: the upstream OpenAI Agents SDK function-calling agent "
            "(vendored) over the AppWorld benchmark. `run` executes the agent and dumps "
            "per-task predictions; `evaluate` scores them with AppWorld's evaluator (TGC/SGC)."
        ),
    )
    parser.add_argument(
        "command",
        choices=("run", "evaluate"),
        help="`run` executes the agent over a split; `evaluate` scores a finished run.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenAI model id (e.g. 'gpt-5.6', 'gpt-4.1'). Overrides --model-config.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to a JSON model config (see configs/). CLI flags override its fields.",
    )
    parser.add_argument(
        "--family",
        choices=("reasoning", "standard"),
        default=None,
        help="Force the capability family. Default: registry lookup / prefix heuristic.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        help="Reasoning effort (none|minimal|low|medium|high|xhigh|max). Ignored for non-reasoning models.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. Ignored for reasoning models.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sampling seed. Ignored for reasoning models.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Per-request output-token cap. Reasoning tokens count toward it, so keep it high for reasoning models.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=f"Cap on agent turns per task. Default {DEFAULT_MAX_STEPS} (upstream's).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "dev", "test_normal", "test_challenge"),
        default="dev",
        help="AppWorld split. `dev` (default) is the smoke/validation slice; test_* are the headline slices.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Run only the first N tasks of the split (smoke runs). Default: all.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help=(
            "AppWorld root directory: data is read from <output-dir>/data and results are "
            "written to <output-dir>/experiments/outputs/<experiment>. Default '.'."
        ),
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment (output folder) name. Default: derived from model + split.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.model is None and args.model_config is None:
        print("Pass --model or --model-config.", file=sys.stderr)
        return 2

    overrides = {
        "model": args.model,
        "family": args.family,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_output_tokens": args.max_output_tokens,
        "max_steps": args.max_steps,
        "split": args.split,
    }
    if args.model_config is not None:
        spec = RunSpec.from_config_file(args.model_config, **overrides)
    else:
        spec = RunSpec(**{k: v for k, v in overrides.items() if v is not None})
    experiment_name = args.experiment_name or spec.experiment_name()

    # Point AppWorld's path store at --output-dir before importing anything
    # heavy from appworld.
    os.environ["APPWORLD_ROOT"] = os.path.abspath(args.output_dir)

    if args.command == "run":
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is not set (see .env.example).", file=sys.stderr)
            return 2
        import asyncio

        from appworld import load_task_ids

        from .vendored.openai_agents.run import run_agent_on_tasks

        task_ids = load_task_ids(spec.split)
        if args.max_tasks is not None:
            task_ids = task_ids[: args.max_tasks]
        runner_config = build_runner_config(spec)
        asyncio.run(
            run_agent_on_tasks(
                experiment_name=experiment_name,
                task_ids=task_ids,
                api_predictor_config=runner_config["api_predictor"],
                agent_config=runner_config["agent"],
                appworld_config=runner_config["appworld"],
                logger_config=runner_config["logger"],
            )
        )
        print(
            f"\nDone. Score it with:\n  uv run appworld-openai-agents evaluate "
            f"--experiment-name {experiment_name} --split {spec.split}"
        )
        return 0

    # evaluate
    if args.max_tasks is not None:
        # Smoke-run scoring: only the first N tasks of the split were run, so
        # only score those (a full-split evaluate would count the rest as failures).
        import json

        from appworld import load_task_ids
        from appworld.evaluator import evaluate_tasks

        task_ids = load_task_ids(spec.split)[: args.max_tasks]
        evaluation_dict = evaluate_tasks(
            task_ids=task_ids,
            experiment_name=experiment_name,
            suppress_errors=True,
            include_details=True,
            save_reports=True,
        )
        print(json.dumps(evaluation_dict["aggregate"], indent=2))
        return 0

    from appworld.evaluator import evaluate_dataset

    evaluate_dataset(
        experiment_name=experiment_name,
        dataset_name=spec.split,
        suppress_errors=True,
        include_details=True,
        aggregate_only=False,
        save_reports=True,
        print_report=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
