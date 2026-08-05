"""CLI entrypoint for the Harvey LAB legal agent.

Run as ``python -m harvey_lab.cli <command> ...``.

Subcommands:
* ``run`` — run the agent on a split's tasks and write the produced
  deliverables to ``--output-dir`` (agent only; needs the task model's
  provider key).
* ``evaluate`` — reuse completed outputs in ``--output-dir``, run and persist
  any missing tasks, then grade every rubric criterion with the batched judge.
* ``fetch`` — materialize a split's task folders into the cache and stop. Needs
  no model key, so it is the hook for warming the cache at environment-setup
  time rather than mid-run (see ``data/fetch.py``).

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
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .agent.agent import HarveyLabAgent, HarveyLabAgentOutput
from .agent.workspace import extract_text, task_source_from_dir
from .config import HarveyLabConfig
from .data.dataset import HarveyLabRecord, load_records, read_split
from .data.fetch import ensure_task_dirs
from .evaluation import (
    ALL_PASS_RATE_FIELD,
    CRITERION_PASS_RATE_FIELD,
    JudgeCallError,
    build_rubric_judge,
    eval_summary,
    evaluate_outputs_on_records,
    write_json,
)


logger = logging.getLogger("harvey_lab")
RUN_OUTPUTS_FILENAME = "run_outputs.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = HarveyLabConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Harvey LAB legal agent: a Stirrup-harnessed agent with a single "
            "`code_exec` tool that reads a task's documents and produces the "
            "requested deliverable files, graded criterion-by-criterion by a "
            "batched LLM judge with deliverable-scoped context."
        ),
    )
    parser.add_argument(
        "command",
        choices=("run", "evaluate", "fetch"),
        help=(
            "`run` executes and saves; `evaluate` resumes saved outputs and grades them; "
            "`fetch` only downloads the split's task folders into the cache."
        ),
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
        help=(
            "Root for the on-demand task cache. Default: "
            "$HARVEY_LAB_CACHE/<commit> or ~/.cache/harvey_lab/<commit>. A custom "
            "path is reused across commits (fetches are commit-checked, so a pin "
            "bump refetches)."
        ),
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
    parser.add_argument(
        "--task-model",
        type=str,
        default=defaults.task_model,
        help="LiteLLM model string for the agent; an `openrouter/…` route needs only OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--task-temperature",
        type=float,
        default=defaults.task_temperature,
        help="Sampling temperature for the agent model.",
    )
    parser.add_argument(
        "--task-reasoning-effort",
        type=str,
        default=defaults.task_reasoning_effort,
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
        help="Reasoning budget for a thinking task model (default xhigh = DeepSeek V4 Pro max reasoning); use none for non-reasoning models.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=defaults.judge_model,
        help="LiteLLM model string for the rubric judge. Keep it fixed when comparing runs.",
    )
    parser.add_argument(
        "--judge-batch-size",
        type=int,
        default=defaults.judge_batch_size,
        help="Criteria graded per judge call (same deliverable scope). Higher is cheaper, 1 is one call per criterion.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=defaults.max_turns,
        help="Cap on the agent's tool-use turns per task (LAB-AA uses 200).",
    )
    parser.add_argument(
        "--task-llm-timeout",
        type=float,
        default=defaults.task_llm_timeout,
        help="Per-LLM-call timeout in seconds for the agent (task) model.",
    )
    parser.add_argument(
        "--judge-llm-timeout",
        type=float,
        default=defaults.judge_llm_timeout,
        help="Per-LLM-call timeout in seconds for the rubric judge (a fast model, so keep it tight).",
    )
    parser.add_argument(
        "--judge-num-retries",
        type=int,
        default=defaults.judge_num_retries,
        help="Retries litellm applies to a failed judge call, so a transient outage does not score criteria FAIL.",
    )
    parser.add_argument(
        "--shell-timeout",
        type=int,
        default=defaults.shell_timeout_s,
        help="Per-`code_exec`-command timeout in seconds (LAB-AA uses 1200).",
    )
    parser.add_argument(
        "--no-view-image",
        action="store_true",
        help="Withhold the `view_image` tool (LAB-AA grants it to vision-capable models).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Maximum agent tasks and grading cases processed concurrently; use 1 for fully sequential execution.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("harvey_lab_run"),
        help="Where deliverables, `run_outputs.json`, and the eval reports are written (and resumed from).",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="With `evaluate`, rerun selected tasks instead of reusing completed saved outputs.",
    )
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> HarveyLabConfig:
    return HarveyLabConfig(
        task_model=args.task_model,
        task_temperature=args.task_temperature,
        task_reasoning_effort=args.task_reasoning_effort,
        judge_model=args.judge_model,
        judge_batch_size=args.judge_batch_size,
        max_turns=args.max_turns,
        task_llm_timeout=args.task_llm_timeout,
        judge_llm_timeout=args.judge_llm_timeout,
        judge_num_retries=args.judge_num_retries,
        shell_timeout_s=args.shell_timeout,
        enable_view_image=not args.no_view_image,
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
        task_source=task_source_from_dir(tasks_root),
    )


def _read_run_manifest(output_dir: Path) -> dict[str, dict[str, Any]]:
    path = output_dir / RUN_OUTPUTS_FILENAME
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list.")
    entries: dict[str, dict[str, Any]] = {}
    for value in payload:
        if not isinstance(value, dict) or not str(value.get("task_id") or ""):
            raise ValueError(f"{path} contains an invalid run entry.")
        entry = dict(value)
        entries[str(entry["task_id"])] = entry
    return entries


def _task_output_dir(output_dir: Path, task_id: str) -> Path:
    root = output_dir.resolve()
    task_dir = (root / task_id).resolve()
    if not task_dir.is_relative_to(root):
        raise ValueError(f"Unsafe task output directory for {task_id}")
    return task_dir


def _artifact_path(output_dir: Path, task_id: str, name: str) -> Path:
    task_dir = _task_output_dir(output_dir, task_id)
    path = (task_dir / name).resolve()
    if not path.is_relative_to(task_dir):
        raise ValueError(f"Unsafe persisted deliverable path for {task_id}: {name}")
    return path


def _is_gradeable(output_dir: Path, record: HarveyLabRecord, entry: dict[str, Any] | None) -> bool:
    """Whether a persisted output can be reloaded from disk and graded.

    A run is gradeable whenever it is not an agent error and its metadata +
    produced files are intact — regardless of *how* it ended. A max-turns or
    partial run is graded on whatever it managed to submit (matching the old
    run-then-grade path), rather than being bucketed as an error.
    """
    if entry is None or "error" in entry:
        return False
    produced = entry.get("deliverables_produced")
    missing = entry.get("deliverables_missing")
    if not isinstance(produced, list) or not isinstance(missing, list):
        return False
    if not all(isinstance(name, str) for name in [*produced, *missing]):
        return False
    expected = set(record.deliverable_names or ("response.md",))
    produced_names = set(produced)
    missing_names = set(missing)
    if produced_names & missing_names or produced_names | missing_names != expected:
        return False
    try:
        return all(_artifact_path(output_dir, record.task_id, name).is_file() for name in produced)
    except (OSError, ValueError):
        return False


def _is_reusable(output_dir: Path, record: HarveyLabRecord, entry: dict[str, Any] | None) -> bool:
    """Whether a saved output is a completed terminal run we can skip re-running.

    Stricter than :func:`_is_gradeable`: only a genuinely finished-or-abandoned
    run whose source-document fingerprint is unchanged is reused. A max-turns
    (or errored / stale-fingerprint) task is re-run before grading.
    """
    if entry is None or not _is_gradeable(output_dir, record, entry):
        return False
    finished = entry.get("finished") is True
    abandoned = entry.get("abandoned") is True
    if finished == abandoned or entry.get("max_turns_reached") is True:
        return False
    return entry.get("task_fingerprint") == record.task_fingerprint


def _persist_agent_output(
    output_dir: Path,
    record: HarveyLabRecord,
    output: HarveyLabAgentOutput,
) -> dict[str, Any]:
    # Write the new deliverables first (atomically), *then* prune whatever this
    # task dir held before — never the reverse. Wiping up front would let a
    # failed write destroy the previous run's files and drop the just-produced
    # output; writing first means a prior run survives a write failure, and the
    # post-write prune still clears stale leftovers (this dir only ever holds
    # this task's deliverables) so a produce-nothing rerun leaves none behind.
    task_dir = _task_output_dir(output_dir, record.task_id)
    written: set[Path] = set()
    for name, content in output.raw_deliverables.items():
        destination = _artifact_path(output_dir, record.task_id, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_bytes(content)
        temporary.replace(destination)
        written.add(destination)
    if task_dir.exists():
        for path in sorted(task_dir.rglob("*"), reverse=True):
            if path.is_file() and path not in written:
                path.unlink()
            elif path.is_dir() and not any(path.iterdir()):
                path.rmdir()
    return {
        "task_id": record.task_id,
        "practice_area": record.practice_area,
        "task_fingerprint": record.task_fingerprint,
        "deliverables_produced": sorted(output.deliverables),
        "deliverables_missing": sorted(output.missing_deliverables),
        "finished": output.finished,
        "abandoned": output.abandoned,
        "max_turns_reached": output.max_turns_reached,
        "total_turns": output.total_turns,
        "wall_seconds": output.wall_seconds,
        "final_answer": output.final_answer,
    }


async def _run_and_persist(
    *,
    agent: HarveyLabAgent,
    records: list[HarveyLabRecord],
    output_dir: Path,
    max_concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def _one(record: HarveyLabRecord) -> dict[str, Any]:
        async with semaphore:
            try:
                output = await agent.forward(record=record)
                return _persist_agent_output(output_dir, record, output)
            except Exception as exc:  # noqa: BLE001 - report, don't abort the batch
                logger.warning("Task %s failed: %s", record.task_id, exc)
                return {
                    "task_id": record.task_id,
                    "practice_area": record.practice_area,
                    "error": f"{type(exc).__name__}: {exc}",
                }

    return list(await asyncio.gather(*[_one(record) for record in records]))


def _merge_and_write_manifest(
    output_dir: Path,
    previous: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged = dict(previous)
    merged.update((str(result["task_id"]), result) for result in results)
    write_json(output_dir / RUN_OUTPUTS_FILENAME, list(merged.values()))
    return merged


def _load_persisted_outputs(
    output_dir: Path,
    records: list[HarveyLabRecord],
    manifest: dict[str, dict[str, Any]],
) -> tuple[dict[str, HarveyLabAgentOutput], dict[str, str]]:
    outputs: dict[str, HarveyLabAgentOutput] = {}
    errors: dict[str, str] = {}
    for record in records:
        entry = manifest.get(record.task_id)
        if entry is None:
            errors[record.task_id] = "No persisted run output is available."
            continue
        if "error" in entry:
            errors[record.task_id] = str(entry["error"])
            continue
        if not _is_gradeable(output_dir, record, entry):
            errors[record.task_id] = "Persisted run metadata or deliverable files are incomplete."
            continue
        produced = [str(name) for name in entry["deliverables_produced"]]
        deliverables = {name: extract_text(_artifact_path(output_dir, record.task_id, name)) for name in produced}
        outputs[record.task_id] = HarveyLabAgentOutput(
            final_answer=str(entry.get("final_answer") or ""),
            deliverables=deliverables,
            missing_deliverables=[str(name) for name in entry["deliverables_missing"]],
            finished=bool(entry.get("finished", False)),
            abandoned=bool(entry.get("abandoned", False)),
            max_turns_reached=bool(entry.get("max_turns_reached", False)),
            total_turns=int(entry.get("total_turns", 0)),
            wall_seconds=float(entry.get("wall_seconds", 0.0)),
        )
    return outputs, errors


def _run_run(args: argparse.Namespace) -> int:
    tasks_root, records = _select_records(args)
    if not records:
        logger.error("run got no tasks for --split %s.", args.split)
        return 2
    config = _config_from_args(args)
    agent = _build_agent(tasks_root, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous = _read_run_manifest(args.output_dir)
    logger.info("Running the agent on %d task(s) from split=%s...", len(records), args.split)
    results = asyncio.run(
        _run_and_persist(
            agent=agent,
            records=records,
            output_dir=args.output_dir,
            max_concurrency=args.max_concurrency,
        )
    )
    _merge_and_write_manifest(args.output_dir, previous, results)
    num_errored = sum(1 for result in results if "error" in result)
    logger.info(
        "Wrote outputs for %d task(s) under %s (%d succeeded, %d errored)",
        len(results),
        args.output_dir,
        len(results) - num_errored,
        num_errored,
    )
    return 0


def _run_fetch(args: argparse.Namespace) -> int:
    """Materialize the selected split's task trees into the cache and stop.

    Downloading is the one part of a run that is slow, network-bound and shared
    across every task, so it is worth doing once up front — from a setup script,
    a warm-up step, or by hand — instead of during evaluation.
    """
    task_ids = read_split(args.split)
    if args.limit is not None:
        task_ids = task_ids[: max(0, args.limit)]
    if not task_ids:
        logger.error("fetch got no tasks for --split %s.", args.split)
        return 2
    tasks_root = ensure_task_dirs(task_ids, cache_dir=args.cache_dir)
    # Loading the records is the cheapest end-to-end proof the trees are usable:
    # it reads every task.json and fingerprints every document on disk.
    records = load_records(tasks_root, task_ids=task_ids)
    logger.info(
        "Materialized %d task(s) of split=%s under %s (%d document(s)).",
        len(records),
        args.split,
        tasks_root,
        sum(len(record.documents) for record in records),
    )
    return 0


def _archive_eval_reports(output_dir: Path) -> None:
    for name in ("eval_summary.json", "eval_outputs.json"):
        current = output_dir / name
        if current.is_file():
            current.replace(output_dir / f"{current.stem}.previous.json")


def _run_evaluate(args: argparse.Namespace) -> int:
    tasks_root, records = _select_records(args)
    if not records:
        logger.error("evaluate got no tasks for --split %s.", args.split)
        return 2
    config = _config_from_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_run_manifest(args.output_dir)
    to_run = [
        record
        for record in records
        if args.rerun or not _is_reusable(args.output_dir, record, manifest.get(record.task_id))
    ]
    if to_run:
        logger.info(
            "Running and saving %d task(s); reusing %d completed output(s)...",
            len(to_run),
            len(records) - len(to_run),
        )
        agent = _build_agent(tasks_root, config)
        results = asyncio.run(
            _run_and_persist(
                agent=agent,
                records=to_run,
                output_dir=args.output_dir,
                max_concurrency=args.max_concurrency,
            )
        )
        manifest = _merge_and_write_manifest(args.output_dir, manifest, results)
    else:
        logger.info("Reusing all %d completed output(s); no agent run needed.", len(records))

    outputs, errors = _load_persisted_outputs(args.output_dir, records, manifest)
    judge = build_rubric_judge(
        model=config.judge_model,
        timeout=config.judge_llm_timeout,
        num_retries=config.judge_num_retries,
    )
    summary_path = args.output_dir / "eval_summary.json"
    outputs_path = args.output_dir / "eval_outputs.json"
    _archive_eval_reports(args.output_dir)
    logger.info(
        "Grading %d task(s) with up to %d concurrent case(s)...",
        len(records),
        max(1, args.max_concurrency),
    )
    try:
        report = asyncio.run(
            evaluate_outputs_on_records(
                records=records,
                outputs=outputs,
                errors=errors,
                judge=judge,
                batch_size=config.judge_batch_size,
                max_concurrency=args.max_concurrency,
            )
        )
    except JudgeCallError as exc:
        logger.error("Evaluation aborted because rubric grading is incomplete: %s", exc)
        return 1
    summary = eval_summary(report, split=args.split)
    write_json(summary_path, summary)
    write_json(outputs_path, report.per_case)
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
    if args.command == "fetch":
        return _run_fetch(args)
    if args.command == "run":
        return _run_run(args)
    return _run_evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
