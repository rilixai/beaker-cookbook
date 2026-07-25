"""CLI entrypoint for HotpotQA benchmarking (SDK-only / Shape B).

Run as ``python -m hotpotqa.cli ...`` after installing this recipe from its
directory (``cd hotpotqa && uv sync --group dev``).

Subcommands:
* ``validate`` — build the spec + run ``validate_spec`` (fully offline; no
  network, no dataset download).
* ``evaluate`` — score ONE candidate (the seed prompts by default, or a
  ``--candidate-json``) on the loaded cases via the SDK ``run_case`` + scorer
  loop, writing an ``eval_summary.json`` + ``eval_outputs.json``.

With no flags, ``evaluate`` runs the seed candidate on the GEPA paper's exact
300-case test slice under the paper's fullwiki / k=7 setup. Data loading is
bit-faithful to the artifact: source is the HotpotQA *train* split; we slice
``[0, 40%)`` for test, ``[40%, 80%)`` for val, ``[80%, 100%)`` for train, then
sample with ``random.Random(1)``. See
:func:`hotpotqa.data.dataset.load_hotpotqa_paper_split` for the full provenance.

The full GEPA optimize loop is intentionally NOT part of this CLI: the
optimizer engine lives in the optional ``rilixai-runtime`` package and runs
server-side for hosted ``rilixai run`` triggers (see ``sandbox.py`` +
``rilixai.yaml``). This recipe depends on the lightweight ``rilixai`` SDK only.

``--no-network`` is the test-friendly guard: instead of downloading the HF
dataset it raises ``RuntimeError`` so a misconfigured run never accidentally
hits HF / an LLM.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

from rilixai import Case, OptimizationTargets

from .agent.prompts import hotpotqa_pydantic_agent_seed_targets
from .config import HotpotQAConfig
from .data.dataset import load_hotpotqa_paper_split
from .optimization.cli_support import load_targets_from_json, validate_and_log, write_eval_report
from .optimization.local_eval import run_local_evaluation
from .optimization.metrics import ANSWER_FIELD
from .optimization.spec import build_hotpotqa_spec


logger = logging.getLogger("hotpotqa")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "HotpotQA benchmark for the rilixai prompt optimizer (SDK-only). "
            "`validate` builds + validates the spec offline; `evaluate` scores "
            "a single candidate on the GEPA paper's held-out split via the SDK "
            "run_case + scorer loop. The full GEPA optimization runs server-side "
            "via `rilixai run`."
        ),
    )
    parser.add_argument(
        "command",
        choices=("validate", "evaluate"),
        help="`validate` builds + validates the spec offline; `evaluate` scores a candidate.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=300,
        help="Validation case count drawn from the paper's [40%%, 80%%) slice. Default 300.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=300,
        help="Test case count drawn from the paper's [0%%, 40%%) slice. Default 300.",
    )
    parser.add_argument(
        "--split",
        choices=("test", "validation"),
        default="test",
        help="`evaluate` only: which paper split to score the candidate on. Default `test`.",
    )
    parser.add_argument(
        "--retrieval",
        choices=("fullwiki", "distractor"),
        default="fullwiki",
        help="Retrieval corpus. `fullwiki` (default, paper parity) or `distractor` (test-friendly).",
    )
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=7,
        help="Paragraphs returned per retrieval call (paper uses k=7).",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=8,
        help="Cap on the agent's tool-use loop length.",
    )
    parser.add_argument(
        "--pydantic-agent-model",
        type=str,
        default=None,
        help="PydanticAI model spec for the outer agent (e.g. 'openai:gpt-4.1-mini').",
    )
    parser.add_argument(
        "--task-model",
        type=str,
        default="openai/gpt-4.1-mini",
        help="Default model for the PydanticAI agent + summarize tool. Slash form is auto-translated.",
    )
    parser.add_argument(
        "--task-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the task LLM. Defaults to 0.0.",
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
        default=Path("hotpotqa_run"),
        help="Directory where results are written.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="HuggingFace datasets cache directory (defaults to the user cache).",
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Refuse to download the HF dataset (test / dry-run guard).",
    )
    return parser.parse_args(argv)


def _fmt_hms(seconds: float) -> str:
    total_seconds = int(round(max(0.0, seconds)))
    hours, rem = divmod(total_seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{sec:02d}"


def _load_eval_cases(args: argparse.Namespace) -> list[Case]:
    """Load the paper split the `evaluate` command scores."""
    if args.no_network:
        raise RuntimeError(
            "Refusing to download the HotpotQA dataset because --no-network was set. "
            "This guard is for dry runs / accidental-spend prevention. For offline "
            "structural validation run: uv run --locked python -m pytest hotpotqa/tests "
            "(scripted FunctionModel, zero network)."
        )
    config = "fullwiki" if args.retrieval == "fullwiki" else "distractor"
    cache_dir = str(args.cache_dir) if args.cache_dir else None
    if args.split == "test":
        if args.test_size <= 0:
            raise ValueError("evaluate --split test requires --test-size > 0.")
        return load_hotpotqa_paper_split("test", max_cases=args.test_size, config=config, cache_dir=cache_dir)
    if args.val_size <= 0:
        raise ValueError("evaluate --split validation requires --val-size > 0.")
    return load_hotpotqa_paper_split("validation", max_cases=args.val_size, config=config, cache_dir=cache_dir)


def _load_targets(path: Path | None) -> OptimizationTargets:
    return load_targets_from_json(path, seed_targets=hotpotqa_pydantic_agent_seed_targets())


def _build_spec_for_args(args: argparse.Namespace) -> Any:
    # Fall back to the shared ``--task-model`` when no explicit agent spec is
    # given. The slash→colon rewrite is handled centrally by ``HotpotQAConfig``
    # (see ``to_pydantic_ai_model``), so no per-call normalization is needed here.
    pydantic_agent_model = args.pydantic_agent_model or args.task_model
    config = HotpotQAConfig(
        retrieval_mode=args.retrieval,
        retrieve_k=args.retrieve_k,
        max_iters=args.max_iters,
        pydantic_agent_model=pydantic_agent_model,
        pydantic_agent_temperature=args.task_temperature,
    )
    return build_hotpotqa_spec(config=config)


def _run_validate(args: argparse.Namespace) -> int:
    spec = _build_spec_for_args(args)
    return validate_and_log(spec, logger=logger)


def _run_evaluate(args: argparse.Namespace) -> int:
    cases = _load_eval_cases(args)
    if not cases:
        logger.error("evaluate command got no cases for --split %s.", args.split)
        return 2
    spec = _build_spec_for_args(args)
    targets = _load_targets(args.candidate_json)
    eval_started = time.monotonic()
    logger.info("Starting evaluate on split=%s (%d cases)...", args.split, len(cases))
    report = run_local_evaluation(
        spec=spec,
        targets=targets,
        cases=cases,
        max_concurrency=args.max_concurrency,
    )
    logger.info("evaluate complete in %s (%d cases)", _fmt_hms(time.monotonic() - eval_started), report.num_cases)
    write_eval_report(report, output_dir=args.output_dir, split=args.split)
    logger.info(
        "Split=%s | %s=%.4f over %d cases",
        args.split,
        ANSWER_FIELD,
        report.field_accuracies.get(ANSWER_FIELD, report.objective),
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
