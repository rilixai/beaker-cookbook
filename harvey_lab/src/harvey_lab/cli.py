"""CLI entrypoint for the Harvey LAB legal agent.

Run as ``python -m harvey_lab.cli <command> ...``.

Subcommands:
* ``run`` — run the agent on a split's tasks and write the produced
  deliverables to ``--output-dir`` (agent only; needs the task model's
  provider key).
* ``evaluate`` — reuse completed outputs in ``--output-dir``, run and persist
  any missing tasks, then grade every rubric criterion with the batched judge.

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
        choices=("run", "evaluate"),
        help="`run` executes and saves; `evaluate` resumes saved outputs and grades them.",
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
    parser.add_argument("--task-model", type=str, default=defaults.task_model)
    parser.add_argument("--task-temperature", type=float, default=defaults.task_temperature)
    parser.add_argument("--judge-model", type=str, default=defaults.judge_model)
    parser.add_argument("--judge-batch-size", type=int, default=defaults.judge_batch_size)
    parser.add_argument("--max-turns", type=int, default=defaults.max_turns)
    parser.add_argument("--llm-timeout", type=float, default=defaults.llm_timeout)
    parser.add_argument("--judge-num-retries", type=int, default=defaults.judge_num_retries)
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
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("harvey_lab_run"))
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
        judge_model=args.judge_model,
        judge_batch_size=args.judge_batch_size,
        max_turns=args.max_turns,
        llm_timeout=args.llm_timeout,
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


def _artifact_path(output_dir: Path, task_id: str, name: str) -> Path:
    root = output_dir.resolve()
    task_dir = (root / task_id).resolve()
    path = (task_dir / name).resolve()
    if not task_dir.is_relative_to(root) or not path.is_relative_to(task_dir):
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
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    for name in (previous or {}).get("deliverables_produced", []):
        if isinstance(name, str):
            try:
                _artifact_path(output_dir, record.task_id, name).unlink(missing_ok=True)
            except ValueError:
                pass
    for name, content in output.raw_deliverables.items():
        destination = _artifact_path(output_dir, record.task_id, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_bytes(content)
        temporary.replace(destination)
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
    previous: dict[str, dict[str, Any]],
    max_concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def _one(record: HarveyLabRecord) -> dict[str, Any]:
        async with semaphore:
            try:
                output = await agent.forward(record=record)
                return _persist_agent_output(output_dir, record, output, previous.get(record.task_id))
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
            previous=previous,
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
                previous=manifest,
                max_concurrency=args.max_concurrency,
            )
        )
        manifest = _merge_and_write_manifest(args.output_dir, manifest, results)
    else:
        logger.info("Reusing all %d completed output(s); no agent run needed.", len(records))

    outputs, errors = _load_persisted_outputs(args.output_dir, records, manifest)
    judge = build_rubric_judge(
        model=config.judge_model,
        timeout=config.llm_timeout,
        num_retries=config.judge_num_retries,
    )
    summary_path = args.output_dir / "eval_summary.json"
    outputs_path = args.output_dir / "eval_outputs.json"
    summary_path.unlink(missing_ok=True)
    outputs_path.unlink(missing_ok=True)
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
    if args.command == "run":
        return _run_run(args)
    return _run_evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
