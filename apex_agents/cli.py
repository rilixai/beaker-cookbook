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
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rilixai import OptimizationTargets, optimization_targets_from_prompts, validate_spec

from .agent.prompts import apex_agents_seed_targets
from .config import ApexAgentsConfig
from .data.dataset import DEFAULT_DOMAIN, load_apex_agents_cases
from .data.world_splits import fixed_val_split, stratified_case_cap
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


def _select_eval_cases(args: argparse.Namespace) -> list[Any]:
    """Build the list of cases the `evaluate` command scores."""
    all_cases = _load_all_cases(args)
    if args.split == "validation":
        _, val_cases, _ = fixed_val_split(
            all_cases,
            n_val_worlds=args.val_worlds,
            val_size=(args.val_size if args.val_size and args.val_size > 0 else None),
            seed=args.seed,
        )
        cases = list(val_cases)
    else:  # "all"
        cases = list(all_cases)
    if args.test_size is not None:
        cases = stratified_case_cap(cases, args.test_size, seed=args.seed)
    return cases


def _load_targets(path: Path | None) -> OptimizationTargets:
    if path is None:
        return apex_agents_seed_targets()
    raw = json.loads(path.read_text())
    # Accept the ``OptimizationTargets`` wire shape (``{"prompts": {...}}``), the
    # legacy ``PromptCandidate`` shape (``{"components": {...}}``) written by the
    # pre-migration optimizer, or a bare ``{name: text}`` mapping.
    if isinstance(raw, dict) and "prompts" in raw:
        prompts = raw["prompts"]
    elif isinstance(raw, dict) and "components" in raw:
        prompts = raw["components"]
    else:
        prompts = raw
    if not isinstance(prompts, dict):
        raise ValueError(f"Candidate JSON at {path} must be an object of prompt name → text.")
    return optimization_targets_from_prompts({str(k): str(v) for k, v in prompts.items()})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


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
    validate_spec(spec)
    logger.info(
        "Spec %r validated: %d seed prompt(s) %s.",
        spec.name,
        len(spec.seed_targets.prompts),
        sorted(spec.seed_targets.prompts),
    )
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    cases = _select_eval_cases(args)
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
    summary = {
        "split": args.split,
        "num_cases": report.num_cases,
        "objective": report.objective,
        "field_accuracies": report.field_accuracies,
        "field_sample_counts": report.field_sample_counts,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "eval_summary.json", summary)
    _write_json(args.output_dir / "eval_outputs.json", report.per_case)
    logger.info(
        "Split=%s | %s=%.4f over %d cases",
        args.split,
        RUBRIC_FIELD,
        report.field_accuracies.get(RUBRIC_FIELD, report.objective),
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
