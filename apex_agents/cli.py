"""CLI entrypoint for the APEX-Agents ReAct toolbelt agent.

Run as ``python -m apex_agents.cli <command> ...``.

Subcommands:
* ``run`` — run the agent over the selected tasks and write its answers to
  ``run_outputs.json`` under ``--output-dir`` (agent only; needs the task
  model's provider key).
* ``evaluate`` — run the agent AND grade every rubric criterion with the LLM
  judge, reporting ``rubric_pass_rate`` into ``eval_summary.json`` +
  ``eval_outputs.json`` (needs the task and judge models' provider keys).

Both download the ``mercor/apex-agents`` dataset from HuggingFace (gated —
request access, then ``huggingface-cli login`` or export ``HF_TOKEN``).
``--no-network`` refuses to download the dataset, build the HF-backed world
factory, or call the judge, so a dry run can never spend money by accident.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from .agent.agent import ApexReActAgent
from .agent.world.world import WorldFactory, WorldFiles
from .config import ApexAgentsConfig
from .data.dataset import DEFAULT_DOMAIN, ApexAgentsRecord, load_apex_agents_records
from .data.world_splits import fixed_val_split, stratified_case_cap
from .evaluation import (
    RUBRIC_FIELD,
    RubricJudge,
    build_rubric_judge,
    eval_summary,
    evaluate_agent_on_records,
    heldout_subset_summary,
    write_json,
)


logger = logging.getLogger("apex_agents")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "APEX-Agents: a faithful ReAct toolbelt agent (prompts seeded verbatim "
            "from Archipelago's reference harness) over the per-task world files, "
            "graded by an LLM rubric judge on professional knowledge-work tasks."
        ),
    )
    parser.add_argument(
        "command",
        choices=("run", "evaluate"),
        help="`run` executes the agent and dumps its answers; `evaluate` also grades them.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=DEFAULT_DOMAIN,
        help='Domain subset to load. Default "Investment Banking"; pass "" for every domain.',
    )
    parser.add_argument(
        "--val-worlds",
        type=int,
        default=2,
        help="Number of WHOLE worlds forming the fixed validation pool.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=20,
        help="Validation task count (stratified across the val worlds). 0/None = all.",
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
        help="'all' uses every loaded task; 'validation' uses the fixed cross-world val pool.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the world-level validation carve + stratified caps.",
    )
    parser.add_argument(
        "--task-model",
        type=str,
        default="openai/gpt-4.1-mini-2025-04-14",
        help="LiteLLM model spec for the ReAct agent.",
    )
    parser.add_argument(
        "--task-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the task LLM. Defaults to 0.0.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini/gemini-3.5-flash",
        help="LiteLLM model spec for the rubric judge (default gemini/gemini-3.5-flash).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=60,
        help="Cap on the ReAct loop. Default 60 (smaller than Archipelago's 250).",
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=3.0,
        help="Cap on the agent's per-task spend (in USD). Default 3.0.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=120.0,
        help="Per-LLM-call timeout in seconds for the agent model AND the rubric judge.",
    )
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("apex_agents_run"),
        help="Directory where results are written.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional HuggingFace cache directory.",
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Refuse to download the dataset, build the HF-backed world factory, or call the judge.",
    )
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> ApexAgentsConfig:
    return ApexAgentsConfig(
        task_model=args.task_model,
        task_temperature=args.task_temperature,
        judge_model=args.judge_model,
        max_steps=args.max_steps,
        cost_limit=args.cost_limit,
        llm_timeout=args.llm_timeout,
    )


def _resolve_world_factory(args: argparse.Namespace) -> WorldFactory:
    """Return the per-task world factory the agent builds its world with."""
    if args.no_network:

        def _refuse(_record: ApexAgentsRecord) -> WorldFiles:
            raise RuntimeError(
                "Refusing to construct an HF-backed world because --no-network was set. "
                "This guard is for dry runs; construct ApexReActAgent directly with your "
                "own world factory for fully offline runs."
            )

        return _refuse
    from .agent.world.world import build_world_factory

    return build_world_factory(cache_dir=str(args.cache_dir) if args.cache_dir else None)


def _resolve_judge(args: argparse.Namespace, config: ApexAgentsConfig) -> RubricJudge:
    """Return the rubric judge — refuses to build the real one under --no-network."""
    if args.no_network:

        def _refuse(_criterion: str, _answer: str, _task: str) -> bool:
            raise RuntimeError(
                "Refusing to call the LLM rubric judge because --no-network was set. "
                "Pass your own judge to evaluate_agent_on_records for offline runs."
            )

        return _refuse
    return build_rubric_judge(model=config.judge_model, timeout=config.llm_timeout)


def _load_all_records(args: argparse.Namespace) -> list[ApexAgentsRecord]:
    if args.no_network:
        raise RuntimeError(
            "Refusing to download the gated HF dataset 'mercor/apex-agents' because "
            "--no-network was set. This guard is for dry runs / accidental-spend "
            "prevention. For an offline check run: uv run python -m pytest (FakeWorld + "
            "scripted model + stub judge, zero network). For real runs, request access at "
            "https://huggingface.co/datasets/mercor/apex-agents then `huggingface-cli "
            "login` (or export HF_TOKEN=...)."
        )
    return load_apex_agents_records(
        # An empty --domain keeps every domain.
        domain=args.domain or None,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )


def _select_records(args: argparse.Namespace) -> tuple[list[ApexAgentsRecord], set[str]]:
    """Return the records to process + the reserved validation world ids.

    The second element is non-empty only for ``--split all``, so the caller can
    report a clean cross-world subset alongside the all-tasks number.
    """
    all_records = _load_all_records(args)
    _, val_records, val_world_ids = fixed_val_split(
        all_records,
        n_val_worlds=args.val_worlds,
        val_size=(args.val_size if args.val_size and args.val_size > 0 else None),
        seed=args.seed,
    )
    excluded_world_ids: set[str] = set()
    if args.split == "validation":
        records = list(val_records)
    else:
        records = list(all_records)
        excluded_world_ids = set(val_world_ids)
    if args.test_size is not None:
        records = stratified_case_cap(records, args.test_size, seed=args.seed)
    return records, excluded_world_ids


def _build_agent(args: argparse.Namespace, config: ApexAgentsConfig) -> ApexReActAgent:
    return ApexReActAgent(
        model_name=config.task_model,
        model_temperature=config.task_temperature,
        max_steps=config.max_steps,
        cost_limit=config.cost_limit,
        max_toolbelt_size=config.max_toolbelt_size,
        max_context_tokens=config.max_context_tokens,
        world_factory=_resolve_world_factory(args),
        llm_timeout=config.llm_timeout,
    )


def _run_run(args: argparse.Namespace) -> int:
    records, _ = _select_records(args)
    if not records:
        logger.error("run got no tasks for --split %s.", args.split)
        return 2
    config = _config_from_args(args)
    agent = _build_agent(args, config)

    async def _run_all() -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(max(1, args.max_concurrency))

        async def _one(record: ApexAgentsRecord) -> dict[str, Any]:
            async with semaphore:
                # Contain per-task failures so one erroring task does not
                # cancel the batch (mirrors evaluate_agent_on_records).
                try:
                    output = await agent.forward(record=record)
                except Exception as exc:  # noqa: BLE001 - report, don't abort
                    logger.warning("Task %s failed: %s", record.task_id, exc)
                    return {
                        "task_id": record.task_id,
                        "world_id": record.world_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                agent_error = str(output.extra.get("error") or "")
                if agent_error:
                    # The agent reports a failed run instead of raising.
                    return {
                        "task_id": record.task_id,
                        "world_id": record.world_id,
                        "error": f"{output.status}: {agent_error}",
                    }
                return {
                    "task_id": record.task_id,
                    "world_id": record.world_id,
                    "status": output.status,
                    "final_answer": output.final_answer,
                    "total_steps": output.total_steps,
                    "total_cost": output.total_cost,
                    "wall_seconds": output.wall_seconds,
                    "resum_count": output.resum_count,
                }

        return list(await asyncio.gather(*[_one(r) for r in records]))

    logger.info("Running the agent on %d task(s)...", len(records))
    results = asyncio.run(_run_all())
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
    records, excluded_world_ids = _select_records(args)
    if not records:
        logger.error("evaluate got no tasks for --split %s.", args.split)
        return 2
    config = _config_from_args(args)
    agent = _build_agent(args, config)
    judge = _resolve_judge(args, config)
    logger.info("Starting evaluate on split=%s (%d tasks)...", args.split, len(records))
    report = asyncio.run(
        evaluate_agent_on_records(
            agent=agent,
            records=records,
            judge=judge,
            max_concurrency=args.max_concurrency,
        )
    )
    summary = eval_summary(report, split=args.split)
    # On a full-dataset eval the headline number includes the reserved
    # cross-world validation pool; also report the CLEAN subset of worlds
    # outside it. Free: same run.
    if args.split == "all":
        summary.update(heldout_subset_summary(report.per_case, excluded_world_ids))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "eval_summary.json", summary)
    write_json(args.output_dir / "eval_outputs.json", report.per_case)
    logger.info(
        "Split=%s | %s=%.4f over %d task(s) (%d scored, %d errored, %d unscoreable)",
        args.split,
        RUBRIC_FIELD,
        report.rubric_pass_rate,
        report.num_cases,
        report.num_scored,
        report.num_errored,
        report.num_unscoreable,
    )
    heldout = summary.get(f"{RUBRIC_FIELD}_heldout")
    if args.split == "all" and heldout is not None:
        logger.info(
            "Clean cross-world held-out: %s=%.4f over %d task(s) whose worlds fall "
            "outside the reserved validation pool.",
            RUBRIC_FIELD,
            heldout,
            summary["num_heldout_cases"],
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
