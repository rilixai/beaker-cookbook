"""CLI entrypoint for the Harvey LAB legal agent.

Run as ``python -m harvey_lab.cli <command> ...``.

Subcommands:
* ``run`` — run the agent on a split's tasks and write the produced
  deliverables to ``--output-dir`` (agent only; needs the task model's
  provider key).
* ``evaluate`` — run the agent AND grade every rubric criterion with the
  batched LLM judge, reporting ``all_pass_rate`` / ``criterion_pass_rate``
  (needs the task model's and the judge model's provider keys).

Tasks come from the frozen ``splits/{train,val,test}.txt`` lists (see
``splits/README.md``). By default only the needed task folders are fetched
from GitHub at ``config.HARVEY_LABS_COMMIT`` into a local cache (see
``data/fetch.py``); pass ``--tasks-root`` to use an existing local
``harvey-labs`` checkout's ``tasks/`` dir instead. ``--limit`` caps how many
of the split's tasks actually run — the lists are ordered so any prefix stays
distribution-representative, so ``--limit`` gives a cheap smoke run.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .agent.agent import HarveyLabAgent
from .agent.workspace import task_source_from_dir
from .config import HarveyLabConfig
from .data.dataset import HarveyLabRecord, load_records, read_split
from .data.fetch import ensure_task_dirs
from .evaluation import (
    ALL_PASS_RATE_FIELD,
    CRITERION_PASS_RATE_FIELD,
    build_rubric_judge,
    eval_summary,
    evaluate_agent_on_records,
    write_json,
)


logger = logging.getLogger("harvey_lab")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = HarveyLabConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Harvey LAB legal agent: a Stirrup-harnessed agent that reads a task's "
            "documents and writes deliverables, graded by an all-pass rubric (a "
            "batched LLM judge with deliverable-scoped context)."
        ),
    )
    parser.add_argument(
        "command",
        choices=("run", "evaluate"),
        help="`run` executes the agent and dumps deliverables; `evaluate` also grades them.",
    )
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=None,
        help=(
            "Path to a local `harvey-labs` checkout's `tasks/` dir. Omit to "
            "fetch only the needed tasks from GitHub at the pinned commit into "
            "a local cache."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Where on-demand-fetched tasks are cached (default: $HARVEY_LAB_CACHE or ~/.cache/harvey_lab).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Which frozen split to run (splits/<split>.txt). Default: test.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N tasks of the split (cheap smoke run). Default: all.",
    )
    parser.add_argument("--task-model", type=str, default=defaults.task_model)
    parser.add_argument("--task-temperature", type=float, default=defaults.task_temperature)
    parser.add_argument("--judge-model", type=str, default=defaults.judge_model)
    parser.add_argument("--judge-batch-size", type=int, default=defaults.judge_batch_size)
    parser.add_argument("--max-turns", type=int, default=defaults.max_turns)
    parser.add_argument("--llm-timeout", type=float, default=defaults.llm_timeout)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("harvey_lab_run"))
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> HarveyLabConfig:
    return HarveyLabConfig(
        task_model=args.task_model,
        task_temperature=args.task_temperature,
        judge_model=args.judge_model,
        judge_batch_size=args.judge_batch_size,
        max_turns=args.max_turns,
        llm_timeout=args.llm_timeout,
    )


def _resolve_tasks_root(args: argparse.Namespace, task_ids: list[str]) -> Path:
    """Use ``--tasks-root`` if given, else fetch the needed tasks into the cache."""
    tasks_root: Path | None = args.tasks_root
    if tasks_root is not None:
        return tasks_root
    return ensure_task_dirs(task_ids, cache_dir=args.cache_dir)


def _select_records(args: argparse.Namespace) -> tuple[Path, list[HarveyLabRecord]]:
    task_ids = read_split(args.split)
    if args.limit is not None:
        task_ids = task_ids[: max(0, args.limit)]
    tasks_root = _resolve_tasks_root(args, task_ids)
    return tasks_root, load_records(tasks_root, task_ids=task_ids)


def _build_agent(tasks_root: Path, config: HarveyLabConfig) -> HarveyLabAgent:
    return HarveyLabAgent(
        config=config,
        task_source=task_source_from_dir(tasks_root, max_document_chars=config.max_document_chars),
    )


def _run_run(args: argparse.Namespace) -> int:
    tasks_root, records = _select_records(args)
    if not records:
        logger.error("run got no tasks for --split %s.", args.split)
        return 2
    config = _config_from_args(args)
    agent = _build_agent(tasks_root, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    async def _run_all() -> list[dict[str, object]]:
        semaphore = asyncio.Semaphore(max(1, args.max_concurrency))

        async def _one(record: HarveyLabRecord) -> dict[str, object]:
            async with semaphore:
                # Contain per-task failures so one erroring task does not
                # cancel the batch (mirrors evaluate_agent_on_records).
                try:
                    output = await agent.forward(record=record)
                except Exception as exc:  # noqa: BLE001 - report, don't abort
                    logger.warning("Task %s failed: %s", record.task_id, exc)
                    return {
                        "task_id": record.task_id,
                        "practice_area": record.practice_area,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                out_dir = args.output_dir / record.task_id
                for name, text in output.deliverables.items():
                    dest = out_dir / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(text, encoding="utf-8")
                return {
                    "task_id": record.task_id,
                    "practice_area": record.practice_area,
                    "deliverables_produced": sorted(output.deliverables),
                    "total_turns": output.total_turns,
                    "wall_seconds": output.wall_seconds,
                }

        return list(await asyncio.gather(*[_one(r) for r in records]))

    logger.info("Running the agent on %d task(s) from split=%s...", len(records), args.split)
    results = asyncio.run(_run_all())
    write_json(args.output_dir / "run_outputs.json", results)
    num_errored = sum(1 for r in results if "error" in r)
    logger.info(
        "Wrote outputs for %d task(s) under %s (%d succeeded, %d errored)",
        len(results),
        args.output_dir,
        len(results) - num_errored,
        num_errored,
    )
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    tasks_root, records = _select_records(args)
    if not records:
        logger.error("evaluate got no tasks for --split %s.", args.split)
        return 2
    config = _config_from_args(args)
    agent = _build_agent(tasks_root, config)
    judge = build_rubric_judge(model=config.judge_model, timeout=config.llm_timeout)
    logger.info("Starting evaluate on split=%s (%d tasks)...", args.split, len(records))
    report = asyncio.run(
        evaluate_agent_on_records(
            agent=agent,
            records=records,
            judge=judge,
            batch_size=config.judge_batch_size,
            max_deliverable_chars=config.max_deliverable_chars,
            max_concurrency=args.max_concurrency,
        )
    )
    summary = eval_summary(report, split=args.split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "eval_summary.json", summary)
    write_json(args.output_dir / "eval_outputs.json", report.per_case)
    logger.info(
        "Split=%s | %s=%.4f %s=%.4f over %d task(s) (%d scored, %d errored, %d unscoreable)",
        args.split,
        ALL_PASS_RATE_FIELD,
        report.all_pass_rate,
        CRITERION_PASS_RATE_FIELD,
        report.criterion_pass_rate,
        report.num_cases,
        report.num_scored,
        report.num_errored,
        report.num_unscoreable,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    if args.command == "run":
        return _run_run(args)
    return _run_evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
