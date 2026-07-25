"""CLI entrypoint for APEX-Agents benchmarking (SDK-only / Shape B).

Run as ``python -m apex_agents.cli ...``.

Subcommands:
* ``validate`` — build the spec + run ``validate_spec`` (fully offline; no
  network, no dataset download).
* ``evaluate`` — score ONE candidate (the seed prompts by default, or a
  ``--candidate-json``) on the loaded cases via the SDK ``run_case`` + scorer
  loop, writing an ``eval_summary.json`` + ``eval_outputs.json``.

The full GEPA optimize/kfold loop is intentionally NOT part of this CLI: the
optimizer engine lives in the optional ``rilixai-runtime`` package and runs
server-side for hosted ``rilixai run`` triggers (see ``sandbox.py`` +
``rilixai.yaml``). This recipe depends on the lightweight ``rilixai`` SDK only.

``--no-network`` is the test-friendly guard: instead of building the real HF
world factory + litellm judge + downloading the gated dataset it raises
``RuntimeError`` so a misconfigured run never accidentally hits HF / an LLM.
Tests construct the spec directly via :func:`build_apex_agents_spec` with an
injected :class:`FakeWorld` factory + stub judge and bypass this CLI entirely.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rilixai import OptimizationTargets

from .agent.prompts import apex_agents_seed_targets
from .config import ApexAgentsConfig
from .data.dataset import DEFAULT_DOMAIN, load_apex_agents_cases
from .data.world_splits import fixed_val_split, stratified_case_cap
from .optimization.cli_support import eval_summary, load_targets_from_json, validate_and_log, write_json
from .optimization.local_eval import run_local_evaluation
from .optimization.metrics import RUBRIC_FIELD
from .optimization.spec import build_apex_agents_spec


logger = logging.getLogger("apex_agents")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "APEX-Agents benchmark for the rilixai prompt optimizer (SDK-only). "
            "Drives a faithful ReAct toolbelt agent (seeded verbatim from "
            "Archipelago's reference prompts) and locally evaluates its three "
            "components (system_prompt, task_template, resum_summary_prompt) on "
            "investment-banking tasks with an LLM rubric judge. The full GEPA "
            "optimization runs server-side via `rilixai run`."
        ),
    )
    parser.add_argument(
        "command",
        choices=("validate", "evaluate"),
        help="`validate` builds + validates the spec offline; `evaluate` scores a candidate.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=DEFAULT_DOMAIN,
        help='Domain subset to load. Default "Investment Banking".',
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
        help="Validation case count (stratified across the val worlds). 0/None = all.",
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
        help="`evaluate` only. 'all' scores the entire domain dataset; 'validation' the fixed val pool.",
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
        help="Cap on the inner agent's per-case spend (in USD). Default 3.0.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=120.0,
        help="Per-LLM-call timeout in seconds for the agent model AND the rubric judge.",
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
        help="Refuse to build the real HF world factory / litellm judge / dataset download.",
    )
    return parser.parse_args(argv)


def _resolve_world_factory(args: argparse.Namespace) -> Callable[[Any], Any]:
    """Return the per-case world factory the run_case uses to build worlds."""
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
    # None → the run_case builds the default litellm-backed judge.
    return None


def _load_all_cases(args: argparse.Namespace) -> list[Any]:
    if args.no_network:
        raise RuntimeError(
            "Refusing to download the gated HF dataset 'mercor/apex-agents' because "
            "--no-network was set. This guard is for dry runs / accidental-spend "
            "prevention. For offline structural validation run: "
            "uv run --locked python -m pytest apex_agents/tests (FakeWorld + scripted "
            "model + stub judge, zero network). For real runs, request access at "
            "https://huggingface.co/datasets/mercor/apex-agents then `huggingface-cli "
            "login` (or export HF_TOKEN=...)."
        )
    return load_apex_agents_cases(
        domain=args.domain,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )


def _select_eval_cases(args: argparse.Namespace) -> tuple[list[Any], set[str]]:
    """Build the cases the `evaluate` command scores + the excluded worlds.

    The second element is the fixed cross-world validation world ids (non-empty
    only for ``--split all``) so ``_run_evaluate`` can report a clean cross-world
    subset alongside the (validation-inclusive) all-cases number.
    """
    all_cases = _load_all_cases(args)
    # The fixed, seed-derived validation worlds. A hosted GEPA run selects
    # candidates against this pool, so the all-cases score is inflated w.r.t.
    # it; carve it out to get a clean cross-world number (see #7).
    _, val_cases, val_world_ids = fixed_val_split(
        all_cases,
        n_val_worlds=args.val_worlds,
        val_size=(args.val_size if args.val_size and args.val_size > 0 else None),
        seed=args.seed,
    )
    excluded_world_ids: set[str] = set()
    if args.split == "validation":
        cases = list(val_cases)
    else:  # "all"
        cases = list(all_cases)
        excluded_world_ids = {str(w) for w in val_world_ids}
    if args.test_size is not None:
        cases = stratified_case_cap(cases, args.test_size, seed=args.seed)
    return cases, excluded_world_ids


def _heldout_subset_summary(per_case: list[dict[str, Any]], excluded_world_ids: set[str]) -> dict[str, Any]:
    """Clean cross-world subset of a full-dataset eval (pure, testable).

    Restricts to cases whose world (``group_key``) is NOT in
    ``excluded_world_ids`` — i.e. worlds outside the reserved cross-world
    validation pool a hosted GEPA run selects candidates against — and averages
    ``rubric_pass_rate`` over only the held-out cases that were actually scored
    (empty-rubric cases omit the field, matching the report's own aggregate).
    """
    held = [r for r in per_case if r.get("group_key") and str(r["group_key"]) not in excluded_world_ids]
    scores = [
        float(fs[RUBRIC_FIELD]) for r in held if isinstance((fs := r.get("field_scores")), dict) and RUBRIC_FIELD in fs
    ]
    return {
        "excluded_world_ids": sorted(excluded_world_ids),
        "num_heldout_cases": len(held),
        "num_heldout_scored": len(scores),
        f"{RUBRIC_FIELD}_heldout": (sum(scores) / len(scores)) if scores else None,
        "note": (
            f"{RUBRIC_FIELD} is over ALL cases incl. the reserved cross-world validation "
            f"pool a hosted GEPA run selects against (validation-inclusive, NOT a clean "
            f"generalization measure). {RUBRIC_FIELD}_heldout is the subset whose worlds "
            f"fall outside that pool."
        ),
    }


def _load_targets(path: Path | None) -> OptimizationTargets:
    return load_targets_from_json(path, seed_targets=apex_agents_seed_targets())


def _build_spec_for_args(args: argparse.Namespace) -> Any:
    config = ApexAgentsConfig(
        task_model=args.task_model,
        task_temperature=args.task_temperature,
        judge_model=args.judge_model,
        max_steps=args.max_steps,
        cost_limit=args.cost_limit,
        llm_timeout=args.llm_timeout,
    )
    return build_apex_agents_spec(
        model_factory=None,
        config=config,
        world_factory=_resolve_world_factory(args),
        judge=_resolve_judge(args),
    )


def _run_validate(args: argparse.Namespace) -> int:
    # Build with the refusing world factory + judge so validation never
    # touches the network; validate_spec only inspects structure.
    args.no_network = True
    spec = _build_spec_for_args(args)
    return validate_and_log(spec, logger=logger)


def _run_evaluate(args: argparse.Namespace) -> int:
    cases, excluded_world_ids = _select_eval_cases(args)
    if not cases:
        logger.error("evaluate command got no cases for --split %s.", args.split)
        return 2
    spec = _build_spec_for_args(args)
    targets = _load_targets(args.candidate_json)
    logger.info("Starting evaluate on split=%s (%d cases)...", args.split, len(cases))
    report = run_local_evaluation(
        spec=spec,
        targets=targets,
        cases=cases,
        max_concurrency=args.max_concurrency,
    )
    summary = eval_summary(report, split=args.split)
    # On a full-dataset eval the score is validation-inclusive (inflated,
    # not a clean generalization measure). Also report the CLEAN cross-world
    # subset — cases outside the reserved validation worlds. Free: same run.
    if args.split == "all":
        summary.update(_heldout_subset_summary(report.per_case, excluded_world_ids))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "eval_summary.json", summary)
    write_json(args.output_dir / "eval_outputs.json", report.per_case)
    logger.info(
        "Split=%s | %s=%.4f over %d cases",
        args.split,
        RUBRIC_FIELD,
        report.field_accuracies.get(RUBRIC_FIELD, report.objective),
        report.num_cases,
    )
    heldout = summary.get(f"{RUBRIC_FIELD}_heldout")
    if args.split == "all" and heldout is not None:
        logger.info(
            "Clean cross-world held-out: %s=%.4f over %d scored cases (worlds outside "
            "the reserved validation pool); the all-cases number above is "
            "validation-inclusive / not a clean generalization measure.",
            RUBRIC_FIELD,
            heldout,
            summary["num_heldout_scored"],
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
