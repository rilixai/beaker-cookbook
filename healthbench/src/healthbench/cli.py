"""CLI entrypoint for the HealthBench recipe (``uv run healthbench <command>``).

Subcommands:
* ``fetch`` — download a subset's JSONL into the cache and stop (no keys).
* ``run`` — answer a split's conversations and write ``run_outputs.json``.
* ``evaluate`` — reuse any responses in ``run_outputs.json``, answer the rest,
  grade every rubric criterion, and write ``eval_summary.json`` (clipped-mean
  scores overall and per theme / axis) plus ``eval_outputs.json`` (per case,
  with every criterion's verdict and the judge's explanation).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from .agent.agent import HealthBenchAgent
from .config import HealthBenchConfig
from .data.dataset import SUBSETS, HealthBenchRecord, fetch_subset, load_records
from .evaluation import OVERALL, RubricGrader, evaluate_records, run_records, write_json


logger = logging.getLogger("healthbench")
RUN_OUTPUTS = "run_outputs.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = HealthBenchConfig()
    parser = argparse.ArgumentParser(description="HealthBench: answer health conversations, grade them by rubric.")
    parser.add_argument("command", choices=("fetch", "run", "evaluate"))
    parser.add_argument("--subset", choices=sorted(SUBSETS), default="full", help="Default `full`.")
    parser.add_argument("--split", choices=("train", "test", "all"), default="test", help="Default `test`.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cases of the split.")
    parser.add_argument("--task-model", default=defaults.task_model)
    parser.add_argument("--task-temperature", type=float, default=defaults.task_temperature)
    parser.add_argument("--task-max-tokens", type=int, default=defaults.task_max_tokens)
    parser.add_argument("--grader-model", default=defaults.grader_model)
    parser.add_argument("--max-concurrency", type=int, default=defaults.max_concurrency)
    parser.add_argument("--output-dir", type=Path, default=Path("healthbench_results"))
    parser.add_argument("--cache-dir", type=Path, default=None, help="Default ~/.cache/healthbench.")
    return parser.parse_args(argv)


def _config(args: argparse.Namespace) -> HealthBenchConfig:
    return HealthBenchConfig(
        task_model=args.task_model,
        task_temperature=args.task_temperature,
        task_max_tokens=args.task_max_tokens,
        grader_model=args.grader_model,
        max_concurrency=args.max_concurrency,
    )


def _load(args: argparse.Namespace) -> list[HealthBenchRecord]:
    records = load_records(args.subset, args.split, limit=args.limit, cache_dir=args.cache_dir)
    logger.info("Loaded %d case(s) from subset=%s split=%s.", len(records), args.subset, args.split)
    return records


def _agent(client: AsyncOpenAI, config: HealthBenchConfig) -> HealthBenchAgent:
    return HealthBenchAgent(
        client, model=config.task_model, temperature=config.task_temperature, max_tokens=config.task_max_tokens
    )


def _previous_outputs(output_dir: Path) -> dict[str, dict[str, Any]]:
    path = output_dir / RUN_OUTPUTS
    if not path.exists():
        return {}
    return {o["prompt_id"]: o for o in json.loads(path.read_text())}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    if args.command == "fetch":
        logger.info("Cached %s", fetch_subset(args.subset, args.cache_dir))
        return 0

    records = _load(args)
    if not records:
        logger.error("No cases for subset=%s split=%s.", args.subset, args.split)
        return 2
    config = _config(args)
    client = AsyncOpenAI()
    agent = _agent(client, config)

    if args.command == "run":
        outputs = asyncio.run(run_records(agent, records, max_concurrency=config.max_concurrency))
        write_json(args.output_dir / RUN_OUTPUTS, outputs)
        logger.info("Wrote %d response(s) to %s.", len(outputs), args.output_dir / RUN_OUTPUTS)
        return 0

    grader = RubricGrader(
        client, model=config.grader_model, max_retries=config.grader_max_retries, concurrency=config.max_concurrency
    )
    outputs, per_case, summary = asyncio.run(
        evaluate_records(
            agent, grader, records, max_concurrency=config.max_concurrency, previous=_previous_outputs(args.output_dir)
        )
    )
    summary = {
        "subset": args.subset,
        "split": args.split,
        "task_model": config.task_model,
        "grader_model": config.grader_model,
        **summary,
    }
    write_json(args.output_dir / RUN_OUTPUTS, outputs)
    write_json(args.output_dir / "eval_outputs.json", per_case)
    write_json(args.output_dir / "eval_summary.json", summary)
    overall = summary["metrics"][OVERALL]
    logger.info(
        "HealthBench %s/%s: %s=%.4f ± %.4f over %d case(s) (%d errored)",
        args.subset,
        args.split,
        OVERALL,
        overall["mean"],
        overall["bootstrap_std"],
        summary["num_cases"],
        summary["num_errored"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
