"""CLI entrypoint for the AppWorld openai-agents baseline.

Run as ``uv run appworld-openai-agents <command> ...`` (or ``python -m
appworld_openai_agents.cli <command> ...``) after installing this project from
its directory (``cd appworld && uv sync --group dev``), and
after ``appworld install`` + ``appworld download data``.

Subcommands:
* ``run`` — run the code agent over a split and write predictions under
  ``<output-dir>/experiments/outputs/<experiment-name>/``.
* ``evaluate`` — score an experiment with AppWorld's evaluator and print
  TGC/SGC (Task / Scenario Goal Completion).

The model is chosen with ``--config <toml>`` (see ``configs/``) or ``--model``.
Whether a model is a reasoning model (GPT-5 / o-series) or a standard one is
inferred from its name; ``--reasoning-effort`` / ``--temperature`` each apply
only to their kind of model and error out if given to the other.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from appworld import load_task_ids
from appworld.common.printer import table_data_to_string
from appworld.evaluator import Metric, evaluate_dataset, evaluate_tasks

from appworld_openai_agents.models import REASONING_EFFORTS, ModelProfile, infer_family
from appworld_openai_agents.runner import MAX_STEPS, run


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AppWorld baseline: a ReAct-style code agent on the OpenAI Agents SDK, with a "
            "capability-aware model layer for sweeping reasoning and non-reasoning OpenAI "
            "models. `run` produces predictions; `evaluate` scores them (TGC/SGC)."
        ),
    )
    parser.add_argument(
        "command",
        choices=("run", "evaluate"),
        help="`run` executes the agent and writes predictions; `evaluate` scores an experiment.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="TOML model config with one table per model (see configs/model.toml); --model picks the table.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenAI model name (with --config: which table to use). Default `gpt-5.6` / the config's `default`.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default=None,
        help="Reasoning effort (reasoning models only). Default `medium`.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (standard models only). Default 0.0.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Per-request output-token cap (reasoning tokens count toward it). Family default if unset.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "dev", "test_normal", "test_challenge"),
        default="dev",
        help="AppWorld dataset split. Default `dev` (the smoke/validation slice).",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Run/evaluate only the first N tasks of the split (smoke runs).",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Run/evaluate a single task id (overrides --split/--max-tasks task selection).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help=f"Agent-loop turn budget per task. Default {MAX_STEPS} (upstream reference).",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment (output folder) name. Default `<agent>_<model>_<split>`.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help=(
            "AppWorld root: predictions go to <output-dir>/experiments/outputs/ and the "
            "benchmark data must live at <output-dir>/data (where `appworld download data` "
            "put it). Default: current directory."
        ),
    )
    return parser.parse_args(argv)


def _profile_from_args(args: argparse.Namespace) -> ModelProfile:
    if args.config is not None:
        return ModelProfile.from_toml(args.config, model=args.model)
    model = args.model or "gpt-5.6"
    family = infer_family(model)
    if family == "reasoning" and args.temperature is not None:
        raise SystemExit(f"--temperature is not supported by reasoning models like {model!r} (the API rejects it).")
    if family == "standard" and args.reasoning_effort is not None:
        raise SystemExit(f"--reasoning-effort is not supported by non-reasoning models like {model!r}.")
    return ModelProfile(
        name=model,
        reasoning_effort=args.reasoning_effort or "medium",
        temperature=args.temperature if args.temperature is not None else 0.0,
        max_output_tokens=args.max_output_tokens,
    )


def _task_ids(args: argparse.Namespace) -> list[str]:
    if args.task_id:
        return [args.task_id]
    task_ids: list[str] = list(load_task_ids(args.split))
    if args.max_tasks is not None:
        task_ids = task_ids[: args.max_tasks]
    return task_ids


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # AppWorld resolves data/ and experiments/outputs/ under APPWORLD_ROOT.
    os.environ["APPWORLD_ROOT"] = str(args.output_dir.resolve())
    profile = _profile_from_args(args)
    experiment_name = args.experiment_name or "_".join(["code", profile.name, args.split])

    if args.command == "run":
        run(
            experiment_name=experiment_name,
            task_ids=_task_ids(args),
            profile=profile,
            max_steps=args.max_steps,
        )
        evaluate_flags = [f"--split {args.split}", f"--experiment-name {experiment_name}"]
        if args.task_id:
            evaluate_flags.append(f"--task-id {args.task_id}")
        elif args.max_tasks is not None:
            evaluate_flags.append(f"--max-tasks {args.max_tasks}")
        if args.output_dir != Path("."):
            evaluate_flags.append(f"--output-dir {args.output_dir}")
        print(
            f"\nPredictions written under {args.output_dir}/experiments/outputs/{experiment_name}/\n"
            f"Score them with:\n  uv run appworld-openai-agents evaluate {' '.join(evaluate_flags)}"
        )
        return 0

    if args.task_id or args.max_tasks is not None:
        # Smoke-slice scoring: evaluate exactly the tasks that were run, not
        # the whole split (unrun tasks would deflate the metrics).
        evaluation_dict = evaluate_tasks(
            task_ids=_task_ids(args),
            experiment_name=experiment_name,
            suppress_errors=True,
            include_details=True,
            save_reports=True,
        )
        print("\nText Evaluation Report:")
        print(table_data_to_string(Metric.build_report(evaluation_dict)))
    else:
        evaluate_dataset(
            experiment_name=experiment_name,
            dataset_name=args.split,
            suppress_errors=True,
            include_details=True,
            aggregate_only=False,
            save_reports=True,
            print_report=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
