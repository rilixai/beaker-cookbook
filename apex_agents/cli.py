"""CLI entrypoint for APEX-Agents benchmarking.

Run as ``python -m rilixai.benchmarks.apex_agents.cli ...``.

Subcommands:
* ``optimize`` — runs GEPA on the train worlds.
* ``evaluate`` — scores a single candidate on the test/validation split.
* ``kfold`` — runs optimize+evaluate for one ``--fold-index`` of the
  world-level k-fold (or a single offline dry path).

Defaults:
* ``--max-metric-calls 200`` — small HF-friendly budget.
* ``--task-temperature 0.0`` — deterministic at the agent's API level.
* ``--max-steps 60`` / ``--cost-limit 3.0`` — caps on the ReAct loop
  (smaller than Archipelago's 250 to bound sales-demo cost).
* ``--judge-model gemini/gemini-2.5-flash`` — Mercor's default rubric
  judge.
* ``--max-concurrency 4``.

``--no-network`` is the test-friendly guard: instead of building the
real HF world factory + litellm judge it raises ``RuntimeError`` so a
misconfigured production run never accidentally hits HF / an LLM.
Tests construct the spec directly via :func:`build_apex_agents_spec`
with an injected :class:`FakeWorld` factory + stub judge and bypass
this CLI entirely.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from rilixai.prompt_optimization.evaluation import (
    evaluate_candidate_on_cases,
    field_accuracy_rows,
    serialize_eval_outputs,
)
from rilixai.prompt_optimization.models import Case, PromptCandidate
from rilixai.prompt_optimization.optimization import extract_best_candidate, summarize_gepa_result_metadata
from rilixai.prompt_optimization.spec import (
    PromptOptimizationRunConfig,
    build_adapter_from_spec,
    run_optimization_from_spec,
)

from .agent.prompts import apex_agents_seed_candidate
from .config import ApexAgentsConfig
from .data.dataset import DEFAULT_DOMAIN, load_apex_agents_cases, world_ids_for_cases
from .data.world_splits import (
    fixed_val_split,
    stratified_case_cap,
    world_held_out_val_split,
    world_level_folds,
)
from .optimization.spec import build_apex_agents_spec


logger = logging.getLogger("rilixai.benchmarks.apex_agents")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "APEX-Agents benchmark for the rilixai prompt optimizer. Drives a "
            "faithful ReAct toolbelt agent (seeded verbatim from Archipelago's "
            "reference prompts) and optimizes its three components "
            "(system_prompt, task_template, resum_summary_prompt) on "
            "investment-banking tasks, evaluating with an LLM rubric judge."
        ),
    )
    parser.add_argument(
        "command",
        choices=("optimize", "evaluate", "kfold"),
        help="`optimize` runs GEPA; `evaluate` scores a candidate; `kfold` runs one fold.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=DEFAULT_DOMAIN,
        help='Domain subset to load. Default "Investment Banking".',
    )
    parser.add_argument("--k", type=int, default=5, help="Number of world-level folds. Default 5.")
    parser.add_argument(
        "--fold-index",
        type=int,
        default=0,
        help="Which fold to run for the `kfold` command (0-based).",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help=(
            "Cap on GEPA training cases — the scaling axis (grows across a "
            "sweep). Drawn from the non-validation worlds. None = full pool."
        ),
    )
    parser.add_argument(
        "--train-size-mode",
        choices=("stratified", "frontslice"),
        default="stratified",
        help=(
            "How --train-size caps the pool. 'stratified' (default): "
            "round-robin across all train worlds so the world set stays wide "
            "at every size (clean 'more data, same worlds' curve). "
            "'frontslice': legacy pool[:n] (worlds collapse at small n)."
        ),
    )
    parser.add_argument(
        "--val-worlds",
        type=int,
        default=2,
        help=(
            "Number of WHOLE worlds forming GEPA's FIXED validation pool — "
            "disjoint from the train worlds, so candidate selection rewards "
            "cross-world transfer (anti-overfit). Constant across a sweep."
        ),
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=20,
        help=(
            "Validation case count — HELD CONSTANT across the sweep, fully "
            "decoupled from --train-size. Cases are stratified across the "
            "val worlds. None/0 = all cases in the val worlds."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help="Optional cap on test/validation cases.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "validation", "test"),
        default="all",
        help=(
            "`evaluate` only. 'all' (default): score on the ENTIRE domain "
            "dataset (leaderboard-shaped; the summary also reports a clean "
            "cross-world held-out subset). 'validation': the fixed val pool. "
            "'test': legacy kfold fold-0 held-out worlds."
        ),
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=200,
        help="GEPA metric-call budget for `optimize`. Default 200 (HF-friendly small budget).",
    )
    parser.add_argument("--reflection-minibatch-size", type=int, default=3)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the optimizer + the world-level k-fold shuffle.",
    )
    parser.add_argument(
        "--mode",
        choices=("agent",),
        default="agent",
        help="Which APEX-Agents pipeline shape to run. Only `agent` is wired.",
    )
    parser.add_argument(
        "--task-model",
        type=str,
        default="openai/gpt-4.1-mini-2025-04-14",
        help="LiteLLM model spec for the inner ReAct agent.",
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
        default="gemini/gemini-2.5-flash",
        help="LiteLLM model spec for the rubric judge. Mercor default gemini/gemini-2.5-flash.",
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
        help="Cap on the inner agent's per-case spend (in USD). Default 3.0.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=120.0,
        help=(
            "Per-LLM-call timeout in seconds for the agent model AND the rubric "
            "judge (litellm `timeout`, bounded retries). Default 120. A hung "
            "provider request fails the case fast instead of wedging the run."
        ),
    )
    parser.add_argument(
        "--reflection-model",
        type=str,
        default="openai/gpt-4.1",
        help="Reflection LM passed to optimize_prompts (provider/model or plain name).",
    )
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument(
        "--candidate-json",
        type=Path,
        default=None,
        help="Path to a candidate JSON for `evaluate` mode (defaults to the seed prompts).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("apex_agents_run"),
        help="Directory where results and reflection artifacts are written.",
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
        help="Refuse to build the real HF world factory / litellm judge — tests and dry runs only.",
    )
    return parser.parse_args(argv)


def _resolve_world_factory(args: argparse.Namespace) -> Callable[[Any], Any]:
    """Return the per-case world factory the spec uses to build worlds."""
    if args.no_network:

        def _refuse(_record: Any) -> Any:
            raise RuntimeError(
                "Refusing to construct an HF-backed world because --no-network was set. "
                "This guard is for tests / dry runs; pass a world factory directly to "
                "build_apex_agents_spec for fully offline runs."
            )

        return _refuse
    from .agent.world.world import build_world_factory

    return build_world_factory(cache_dir=str(args.cache_dir) if args.cache_dir else None)


def _resolve_judge(args: argparse.Namespace) -> Callable[[str, str, str], bool] | None:
    """Return the rubric judge — refuses to build the real one under --no-network."""
    if args.no_network:

        def _refuse(_criterion: str, _answer: str, _task: str) -> bool:
            raise RuntimeError(
                "Refusing to call the LLM rubric judge because --no-network was set. "
                "Pass a stub judge directly to build_apex_agents_spec for offline runs."
            )

        return _refuse
    # None → the runtime builds the default litellm-backed judge.
    return None


def _load_all_cases(args: argparse.Namespace) -> list[Case]:
    # The dataset loader is a network call (gated HF dataset
    # ``mercor/apex-agents``). Honor ``--no-network`` here too so a dry
    # run fails fast with a clear message instead of leaking the HF
    # client's "gated repo" traceback. The guard's contract is "tests
    # and dry runs only" — that has to include the dataset download,
    # not just the world factory + judge. Offline structural validation
    # is the test suite (FakeWorld, no HF access required), not this
    # CLI path.
    if args.no_network:
        raise RuntimeError(
            "Refusing to download the gated HF dataset 'mercor/apex-agents' because "
            "--no-network was set. This guard is for dry runs / accidental-spend "
            "prevention. For offline structural validation run: "
            "uv run --locked python -m pytest tests/test_apex_agents_*.py "
            "(FakeWorld + scripted model + stub judge, zero network). For real "
            "runs, request access at https://huggingface.co/datasets/mercor/apex-agents "
            "then `huggingface-cli login` (or export HF_TOKEN=...)."
        )
    return load_apex_agents_cases(
        domain=args.domain,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )


def _split_cases_by_world(
    cases: list[Case],
    *,
    train_world_ids: list[str],
    test_world_ids: list[str],
) -> tuple[list[Case], list[Case]]:
    train_set = set(train_world_ids)
    test_set = set(test_world_ids)
    train = [c for c in cases if str(c.metadata.get("world_id")) in train_set]
    test = [c for c in cases if str(c.metadata.get("world_id")) in test_set]
    return train, test


def _carve_inner_val(train: list[Case], args: argparse.Namespace) -> tuple[list[Case], list[Case]]:
    """Split a fold's train pool into (inner_train, validation) by WHOLE worlds.

    GEPA selects candidates on the validation score; a same-world random
    slice rewards in-world fit and the chosen prompt collapses on unseen
    worlds. World-held-out validation makes selection reward cross-world
    transfer. ``--val-size`` is retained only as an optional cap on the
    number of validation cases (the carving is by world, via
    :func:`world_held_out_val_split`).
    """
    inner_train, validation = world_held_out_val_split(train, n_val_worlds=args.val_worlds, seed=args.seed)
    if args.val_size is not None and args.val_size > 0 and len(validation) > args.val_size:
        validation = validation[: args.val_size]
    return inner_train, validation


def _load_splits_for_command(args: argparse.Namespace) -> dict[str, list[Case]]:
    """Build the cases-by-split mapping for the active command."""
    all_cases = _load_all_cases(args)
    world_ids = world_ids_for_cases(all_cases)
    folds = world_level_folds(world_ids, k=args.k, seed=args.seed)
    if args.command == "kfold":
        if not 0 <= args.fold_index < len(folds):
            raise ValueError(f"--fold-index {args.fold_index} out of range for k={args.k} ({len(folds)} folds).")
        train_world_ids, test_world_ids = folds[args.fold_index]
        train, test = _split_cases_by_world(all_cases, train_world_ids=train_world_ids, test_world_ids=test_world_ids)
        if args.train_size is not None:
            train = train[: args.train_size]
        if args.test_size is not None:
            test = test[: args.test_size]
        inner_train, validation = _carve_inner_val(train, args)
        return {"train": inner_train, "validation": validation, "test": test}

    # optimize / evaluate: FIXED cross-world validation (constant across a
    # train-size sweep) + a growing train pool from the non-val worlds +
    # final eval on the ENTIRE dataset.
    train_pool, val_cases, val_world_ids = fixed_val_split(
        all_cases,
        n_val_worlds=args.val_worlds,
        val_size=(args.val_size if args.val_size and args.val_size > 0 else None),
        seed=args.seed,
    )
    train = stratified_case_cap(train_pool, args.train_size, mode=args.train_size_mode, seed=args.seed)
    train_world_ids = {str(getattr(c, "group_key", "") or "") for c in train}
    # Worlds GEPA saw (trained or validated on) — used to carve a clean
    # cross-world held-out subset from the full-dataset eval.
    args._excluded_world_ids = set(val_world_ids) | train_world_ids  # type: ignore[attr-defined]

    splits: dict[str, list[Case]] = {}
    if args.command == "optimize":
        splits["train"] = train
        splits["validation"] = val_cases
        return splits
    # evaluate
    if args.split == "all":
        splits["all"] = list(all_cases)
    elif args.split == "validation":
        splits["validation"] = val_cases
    else:  # legacy "test": fold-0 held-out worlds
        _, test = _split_cases_by_world(all_cases, train_world_ids=folds[0][0], test_world_ids=folds[0][1])
        if args.test_size is not None:
            test = test[: args.test_size]
        splits["test"] = test
    return splits


def _load_candidate(path: Path | None) -> PromptCandidate:
    if path is None:
        return apex_agents_seed_candidate()
    raw = json.loads(path.read_text())
    return PromptCandidate.from_dict(raw)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _format_field_table(accuracies: dict[str, float], sample_counts: dict[str, int]) -> str:
    rows = field_accuracy_rows(accuracies, sample_counts)
    if not rows:
        return "(no field scores)"
    return "\n".join(f"  {name:<32s} acc={acc:.4f}  n={count}" for name, acc, count in rows)


def _wrap_runtime_with_progress(
    runtime: Callable[..., Awaitable[Any]],
    *,
    label: str,
    total: int | None = None,
    log_every_n: int = 1,
    log_every_seconds: float = 30.0,
) -> Callable[..., Awaitable[Any]]:
    """Wrap an async extraction runtime to log per-case progress + ETA.

    Mirrors the IFBench / HotpotQA / SWE-bench CLI helpers. CRITICAL:
    never ``return`` from the ``finally`` block — that would silently
    override the runtime's actual result with ``None``.
    """
    state: dict[str, Any] = {
        "done": 0,
        "started": None,
        "last_log_count": 0,
        "last_log_time": 0.0,
    }
    lock = threading.Lock()

    async def _wrapped(**kwargs: Any) -> Any:
        with lock:
            if state["started"] is None:
                state["started"] = time.monotonic()
        try:
            result = await runtime(**kwargs)
        finally:
            # Do NOT ``return`` here — see the matching note in
            # swe_bench/cli.py for context.
            with lock:
                state["done"] += 1
                done = state["done"]
                now = time.monotonic()
                elapsed = now - state["started"]
                is_first = done == 1
                is_last = total is not None and done >= total
                interval_ok = (done - state["last_log_count"]) >= log_every_n
                time_ok = (now - state["last_log_time"]) >= log_every_seconds
                should_log = is_first or is_last or interval_ok or time_ok
                if should_log:
                    state["last_log_count"] = done
                    state["last_log_time"] = now
            if should_log:
                rate = (done / elapsed) if elapsed > 0 else 0.0
                if total:
                    remaining = max(0, total - done)
                    eta = (remaining / rate) if rate > 0 else 0.0
                    logger.info(
                        "[%s] %d/%d (%.1f%%) elapsed=%s rate=%.2f/s eta=%s",
                        label,
                        done,
                        total,
                        100.0 * done / total,
                        _fmt_hms(elapsed),
                        rate,
                        _fmt_hms(eta),
                    )
                else:
                    logger.info(
                        "[%s] %d done elapsed=%s rate=%.2f/s",
                        label,
                        done,
                        _fmt_hms(elapsed),
                        rate,
                    )
        return result

    return _wrapped


def _fmt_hms(seconds: float) -> str:
    total_seconds = int(round(max(0.0, seconds)))
    hours, rem = divmod(total_seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{sec:02d}"


def _build_spec_for_args(args: argparse.Namespace, splits: dict[str, list[Case]]) -> Any:
    config = ApexAgentsConfig(
        task_model=args.task_model,
        task_temperature=args.task_temperature,
        judge_model=args.judge_model,
        max_steps=args.max_steps,
        cost_limit=args.cost_limit,
        llm_timeout=args.llm_timeout,
    )
    return build_apex_agents_spec(
        cases_by_split={name: list(cases) for name, cases in splits.items()},
        model=args.task_model,
        max_concurrency=args.max_concurrency,
        config=config,
        world_factory=_resolve_world_factory(args),
        judge=_resolve_judge(args),
    )


def _run_optimize(args: argparse.Namespace, spec: Any, output_dir: Path) -> int:
    run_config = PromptOptimizationRunConfig.from_spec(
        spec,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        reflection_model=args.reflection_model,
        run_dir=str(output_dir),
        reflection_artifact_dir=str(output_dir / "reflection_artifacts"),
        seed=args.seed,
    )
    optimize_started = time.monotonic()
    logger.info("Starting optimize (max_metric_calls=%d)...", args.max_metric_calls)
    result = run_optimization_from_spec(spec, run_config)
    logger.info("optimize complete in %s", _fmt_hms(time.monotonic() - optimize_started))
    best = extract_best_candidate(result)
    metadata = summarize_gepa_result_metadata(result)
    _write_json(output_dir / "best_candidate.json", best.to_dict())
    _write_json(output_dir / "gepa_metadata.json", metadata)
    logger.info("Best candidate written to %s", output_dir / "best_candidate.json")
    logger.info("GEPA metadata: %s", metadata)
    return 0


def _heldout_subset_summary(serialized_rows: list[dict[str, Any]], excluded_world_ids: set[str]) -> dict[str, Any]:
    """Clean cross-world subset of a full-dataset eval (pure, testable).

    Returns the mean ``rubric_pass_rate`` over only the cases whose world
    is NOT in ``excluded_world_ids`` (i.e. worlds GEPA never trained or
    validated on) — the leaderboard-defensible number alongside the
    train-inclusive all-cases number.
    """

    def _world_of(row: dict[str, Any]) -> str:
        gt = row.get("ground_truth") or {}
        if gt.get("world_id"):
            return str(gt["world_id"])
        ap = ((row.get("prediction") or {}).get("run_metrics") or {}).get("apex_agents") or {}
        return str(ap.get("world_id") or "")

    held = [r for r in serialized_rows if _world_of(r) and _world_of(r) not in excluded_world_ids]
    scores = [float((r.get("field_scores") or {}).get("rubric_pass_rate") or 0.0) for r in held]
    return {
        "excluded_world_ids": sorted(excluded_world_ids),
        "num_heldout_cases": len(held),
        "rubric_pass_rate_heldout": (sum(scores) / len(scores)) if scores else None,
        "note": (
            "rubric_pass_rate is over ALL cases incl. those GEPA trained/validated on "
            "(train-inclusive, NOT leaderboard-comparable). rubric_pass_rate_heldout is "
            "the clean cross-world subset (worlds GEPA never saw)."
        ),
    }


def _run_evaluate(args: argparse.Namespace, spec: Any, splits: dict[str, list[Case]], output_dir: Path) -> int:
    adapter = build_adapter_from_spec(spec)
    candidate = _load_candidate(args.candidate_json)
    target_cases = list(splits.get(args.split, []))
    if not target_cases:
        logger.error("evaluate command got no cases for --split %s.", args.split)
        return 2
    eval_started = time.monotonic()
    logger.info("Starting evaluate on split=%s (%d cases)...", args.split, len(target_cases))
    report = evaluate_candidate_on_cases(adapter=adapter, candidate=candidate, cases=target_cases)
    eval_elapsed = time.monotonic() - eval_started
    logger.info("evaluate complete in %s (%d cases)", _fmt_hms(eval_elapsed), len(target_cases))
    serialized = serialize_eval_outputs(report.outputs)
    summary: dict[str, Any] = {
        "split": args.split,
        "num_cases": len(target_cases),
        "weighted_objective": report.weighted_objective,
        "field_accuracies": report.field_accuracies,
        "field_sample_counts": report.field_sample_counts,
    }
    # On a full-dataset eval, the score is train-inclusive (inflated, not
    # leaderboard-comparable). Also report the CLEAN cross-world subset:
    # cases whose world GEPA never trained or validated on. Free — same run.
    if args.split == "all":
        summary.update(_heldout_subset_summary(serialized, getattr(args, "_excluded_world_ids", set()) or set()))
    _write_json(output_dir / "eval_summary.json", summary)
    _write_json(output_dir / "eval_outputs.json", serialized)
    logger.info(
        "Split=%s | weighted_objective=%.4f over %d cases",
        args.split,
        report.weighted_objective,
        len(target_cases),
    )
    logger.info("Field accuracies:\n%s", _format_field_table(report.field_accuracies, report.field_sample_counts))
    if args.split == "all" and summary.get("rubric_pass_rate_heldout") is not None:
        logger.info(
            "Clean cross-world held-out: rubric_pass_rate=%.4f over %d cases "
            "(worlds GEPA never trained/validated on); the all-cases number above "
            "is train-inclusive / not leaderboard-comparable.",
            summary["rubric_pass_rate_heldout"],
            summary["num_heldout_cases"],
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    splits = _load_splits_for_command(args)
    spec = _build_spec_for_args(args, splits)

    if args.command == "evaluate":
        progress_total: int | None = len(splits.get(args.split, []))
        progress_label = f"evaluate:{args.split}"
        progress_every_n = 1
    elif args.command == "kfold":
        progress_total = args.max_metric_calls
        progress_label = f"kfold:{args.fold_index}"
        progress_every_n = max(1, args.max_metric_calls // 80)
    else:
        progress_total = args.max_metric_calls
        progress_label = "optimize"
        progress_every_n = max(1, args.max_metric_calls // 80)
    spec = dataclasses.replace(
        spec,
        extraction_runtime=_wrap_runtime_with_progress(
            spec.extraction_runtime,
            label=progress_label,
            total=progress_total,
            log_every_n=progress_every_n,
            log_every_seconds=30.0,
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "optimize":
        return _run_optimize(args, spec, args.output_dir)

    if args.command == "kfold":
        fold_dir = args.output_dir / f"fold{args.fold_index}"
        (fold_dir / "optimize").mkdir(parents=True, exist_ok=True)
        opt_code = _run_optimize(args, spec, fold_dir / "optimize")
        if opt_code != 0:
            return opt_code
        # Re-evaluate the optimized candidate on the held-out test worlds.
        eval_spec = _build_spec_for_args(args, splits)
        args.candidate_json = fold_dir / "optimize" / "best_candidate.json"
        args.split = "test"
        return _run_evaluate(args, eval_spec, splits, fold_dir / "after")

    # evaluate
    return _run_evaluate(args, spec, splits, args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
