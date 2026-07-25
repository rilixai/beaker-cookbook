"""CLI entrypoint for the Harvey LAB legal agent.

Run as ``python -m harvey_lab.cli <command> ...``.

Subcommands:
* ``run`` — run the agent on tasks from a local ``harvey-labs`` checkout and
  write the produced deliverables to ``--output-dir`` (agent only; needs the
  task model's provider key).
* ``evaluate`` — run the agent AND grade every rubric criterion with the LLM
  judge, reporting ``all_pass`` / ``criterion_pass_rate`` (needs the task
  model's and the judge model's provider keys).

Both load tasks from ``--tasks-root`` (clone
https://github.com/harveyai/harvey-labs and point at its ``tasks/`` dir).
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
from .data.dataset import HarveyLabRecord, load_harvey_lab_records
from .data.task_splits import fixed_val_split, stratified_case_cap
from .evaluation import (
    ALL_PASS_FIELD,
    CRITERION_PASS_RATE_FIELD,
    build_criterion_judge,
    eval_summary,
    evaluate_agent_on_records,
    heldout_subset_summary,
    write_json,
)


logger = logging.getLogger("harvey_lab")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Harvey LAB legal agent: a Stirrup-harnessed agent that reads a task's "
            "documents and writes deliverables, graded by an all-pass rubric (a "
            "per-criterion LLM judge with deliverable-scoped context)."
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
        required=True,
        help="Path to a local `harvey-labs` checkout's `tasks/` dir.",
    )
    parser.add_argument(
        "--practice-areas",
        type=str,
        default=None,
        help="Comma-separated practice areas to load (default: all).",
    )
    parser.add_argument(
        "--max-per-area",
        type=int,
        default=None,
        help="Optional cap on tasks loaded per practice area.",
    )
    parser.add_argument(
        "--val-areas",
        type=int,
        default=2,
        help="Number of WHOLE practice areas forming the fixed validation pool.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=20,
        help="Validation case count (stratified across the val areas). 0/None = all.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help="Optional cap on the number of tasks run/evaluated.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "validation"),
        default="all",
        help="'all' uses every loaded task; 'validation' uses the fixed cross-area val pool.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the area-level validation carve + caps.")
    parser.add_argument("--task-model", type=str, default="openai/gpt-4.1-mini-2025-04-14")
    parser.add_argument("--task-temperature", type=float, default=0.0)
    parser.add_argument("--judge-model", type=str, default="gemini/gemini-3.5-flash")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("harvey_lab_run"))
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> HarveyLabConfig:
    return HarveyLabConfig(
        task_model=args.task_model,
        task_temperature=args.task_temperature,
        judge_model=args.judge_model,
        max_turns=args.max_turns,
        llm_timeout=args.llm_timeout,
    )


def _select_records(args: argparse.Namespace) -> tuple[list[HarveyLabRecord], set[str]]:
    areas = [a.strip() for a in args.practice_areas.split(",")] if args.practice_areas else None
    all_records = load_harvey_lab_records(args.tasks_root, practice_areas=areas, max_per_area=args.max_per_area)
    _, val_records, val_area_ids = fixed_val_split(
        all_records,
        n_val_areas=args.val_areas,
        val_size=(args.val_size if args.val_size and args.val_size > 0 else None),
        seed=args.seed,
    )
    excluded_area_ids: set[str] = set()
    if args.split == "validation":
        records = list(val_records)
    else:
        records = list(all_records)
        excluded_area_ids = {str(a) for a in val_area_ids}
    if args.test_size is not None:
        records = stratified_case_cap(records, args.test_size, seed=args.seed)
    return records, excluded_area_ids


def _build_agent(args: argparse.Namespace, config: HarveyLabConfig) -> HarveyLabAgent:
    return HarveyLabAgent(
        config=config,
        task_source=task_source_from_dir(args.tasks_root, max_document_chars=config.max_document_chars),
    )


def _run_run(args: argparse.Namespace) -> int:
    records, _ = _select_records(args)
    if not records:
        logger.error("run got no tasks for --split %s.", args.split)
        return 2
    config = _config_from_args(args)
    agent = _build_agent(args, config)
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

    logger.info("Running the agent on %d task(s)...", len(records))
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
    records, excluded_area_ids = _select_records(args)
    if not records:
        logger.error("evaluate got no tasks for --split %s.", args.split)
        return 2
    config = _config_from_args(args)
    agent = _build_agent(args, config)
    judge = build_criterion_judge(model=config.judge_model, timeout=config.llm_timeout)
    logger.info("Starting evaluate on split=%s (%d tasks)...", args.split, len(records))
    report = asyncio.run(
        evaluate_agent_on_records(
            agent=agent,
            records=records,
            judge=judge,
            max_deliverable_chars=config.max_deliverable_chars,
            max_concurrency=args.max_concurrency,
        )
    )
    summary = eval_summary(report, split=args.split)
    if args.split == "all":
        summary.update(heldout_subset_summary(report.per_case, excluded_area_ids))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "eval_summary.json", summary)
    write_json(args.output_dir / "eval_outputs.json", report.per_case)
    logger.info(
        "Split=%s | %s=%.4f %s=%.4f over %d task(s) (%d scored, %d errored, %d unscoreable)",
        args.split,
        ALL_PASS_FIELD,
        report.all_pass,
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
