"""CLI entrypoint for Harvey LAB benchmarking (SDK-only / Shape B).

Run as ``python -m harvey_lab.cli ...``.

Subcommands:
* ``validate`` — build the spec + run ``validate_spec`` (fully offline; no
  network, no documents, no LLM).
* ``evaluate`` — score ONE candidate (the seed prompts by default, or a
  ``--candidate-json``) over cases loaded from a local ``harvey-labs``
  checkout (``--tasks-root``), via the SDK ``run_case`` + scorer loop.

``evaluate`` runs the real Stirrup agent + LLM rubric judge, so it needs
provider credentials and spends tokens. The full GEPA optimize/kfold loop
is NOT part of this CLI — it runs server-side for hosted ``rilixai run``
triggers (see ``sandbox.py``). Tests construct the spec directly via
:func:`build_harvey_lab_spec` with a fixture task source + scripted model +
stub judge and bypass this CLI entirely.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from rilixai import OptimizationTargets

from cookbook_common.cli_support import eval_summary, load_targets_from_json, validate_and_log, write_json
from cookbook_common.local_eval import run_local_evaluation

from .agent.prompts import harvey_lab_seed_targets
from .agent.workspace import task_source_from_dir
from .config import HarveyLabConfig
from .data.dataset import cases_from_records, load_harvey_lab_records
from .data.task_splits import fixed_val_split, stratified_case_cap
from .optimization.scoring import ALL_PASS_FIELD, CRITERION_PASS_RATE_FIELD
from .optimization.spec import build_harvey_lab_spec


logger = logging.getLogger("harvey_lab")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Harvey LAB benchmark for the rilixai prompt optimizer (SDK-only). "
            "Drives a Stirrup-harnessed legal agent that reads task documents and "
            "writes deliverables, graded by an all-pass rubric (a per-criterion "
            "LLM judge with deliverable-scoped context). GEPA runs server-side."
        ),
    )
    parser.add_argument(
        "command",
        choices=("validate", "evaluate"),
        help="`validate` builds + validates the spec offline; `evaluate` scores a candidate.",
    )
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=None,
        help="Path to a local `harvey-labs` checkout's `tasks/` dir (required for `evaluate`).",
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
        help="Optional cap on the number of evaluated cases.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "validation"),
        default="all",
        help="`evaluate` only. 'all' scores every loaded case; 'validation' the fixed val pool.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the area-level validation carve + caps.")
    parser.add_argument("--task-model", type=str, default="openai/gpt-4.1-mini-2025-04-14")
    parser.add_argument("--task-temperature", type=float, default=0.0)
    parser.add_argument("--judge-model", type=str, default="gemini/gemini-3.5-flash")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--candidate-json", type=Path, default=None)
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


def _load_all_cases(args: argparse.Namespace) -> list[Any]:
    if args.tasks_root is None:
        raise RuntimeError(
            "evaluate requires --tasks-root pointing at a local harvey-labs checkout's tasks/ dir. "
            "Clone https://github.com/harveyai/harvey-labs and pass its tasks/ path."
        )
    areas = [a.strip() for a in args.practice_areas.split(",")] if args.practice_areas else None
    records = load_harvey_lab_records(args.tasks_root, practice_areas=areas, max_per_area=args.max_per_area)
    return cases_from_records(records)


def _select_eval_cases(args: argparse.Namespace) -> tuple[list[Any], set[str]]:
    all_cases = _load_all_cases(args)
    _, val_cases, val_area_ids = fixed_val_split(
        all_cases,
        n_val_areas=args.val_areas,
        val_size=(args.val_size if args.val_size and args.val_size > 0 else None),
        seed=args.seed,
    )
    excluded_area_ids: set[str] = set()
    if args.split == "validation":
        cases = list(val_cases)
    else:
        cases = list(all_cases)
        excluded_area_ids = {str(a) for a in val_area_ids}
    if args.test_size is not None:
        cases = stratified_case_cap(cases, args.test_size, seed=args.seed)
    return cases, excluded_area_ids


def _heldout_subset_summary(per_case: list[dict[str, Any]], excluded_area_ids: set[str]) -> dict[str, Any]:
    """Clean cross-practice-area subset of a full eval (pure, testable)."""
    held = [r for r in per_case if r.get("group_key") and str(r["group_key"]) not in excluded_area_ids]
    scores = [
        float(fs[ALL_PASS_FIELD])
        for r in held
        if isinstance((fs := r.get("field_scores")), dict) and ALL_PASS_FIELD in fs
    ]
    return {
        "excluded_practice_areas": sorted(excluded_area_ids),
        "num_heldout_cases": len(held),
        "num_heldout_scored": len(scores),
        f"{ALL_PASS_FIELD}_heldout": (sum(scores) / len(scores)) if scores else None,
        "note": (
            f"{ALL_PASS_FIELD} is over ALL cases incl. the reserved cross-area validation "
            f"pool a hosted GEPA run selects against (validation-inclusive). "
            f"{ALL_PASS_FIELD}_heldout is the subset whose practice areas fall outside that pool."
        ),
    }


def _load_targets(path: Path | None) -> OptimizationTargets:
    return load_targets_from_json(path, seed_targets=harvey_lab_seed_targets())


def _build_spec_for_args(args: argparse.Namespace, *, offline: bool) -> Any:
    config = _config_from_args(args)
    if offline:

        def _refuse_source(_record: Any) -> Any:
            raise RuntimeError("Refusing to materialize a task workspace during offline validate.")

        def _refuse_judge(_t: str, _ct: str, _mc: str, _out: str) -> bool:
            raise RuntimeError("Refusing to call the LLM judge during offline validate.")

        return build_harvey_lab_spec(config=config, task_source=_refuse_source, judge=_refuse_judge)
    tasks_root = args.tasks_root
    if tasks_root is None:
        raise RuntimeError("evaluate requires --tasks-root.")
    return build_harvey_lab_spec(
        config=config,
        task_source=task_source_from_dir(tasks_root, max_document_chars=config.max_document_chars),
    )


def _run_validate(args: argparse.Namespace) -> int:
    spec = _build_spec_for_args(args, offline=True)
    return validate_and_log(spec, logger=logger)


def _run_evaluate(args: argparse.Namespace) -> int:
    cases, excluded_area_ids = _select_eval_cases(args)
    if not cases:
        logger.error("evaluate command got no cases for --split %s.", args.split)
        return 2
    spec = _build_spec_for_args(args, offline=False)
    targets = _load_targets(args.candidate_json)
    logger.info("Starting evaluate on split=%s (%d cases)...", args.split, len(cases))
    report = run_local_evaluation(spec=spec, targets=targets, cases=cases, max_concurrency=args.max_concurrency)
    summary = eval_summary(report, split=args.split)
    if args.split == "all":
        summary.update(_heldout_subset_summary(report.per_case, excluded_area_ids))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "eval_summary.json", summary)
    write_json(args.output_dir / "eval_outputs.json", report.per_case)
    logger.info(
        "Split=%s | %s=%.4f %s=%.4f over %d cases",
        args.split,
        ALL_PASS_FIELD,
        report.field_accuracies.get(ALL_PASS_FIELD, 0.0),
        CRITERION_PASS_RATE_FIELD,
        report.field_accuracies.get(CRITERION_PASS_RATE_FIELD, report.objective),
        report.num_cases,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    if args.command == "validate":
        return _run_validate(args)
    return _run_evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
