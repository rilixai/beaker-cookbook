"""CLI entrypoint for HotpotQA benchmarking (SDK-only / Shape B).

Run as ``python -m hotpotqa.cli ...`` after installing the cookbook's
``hotpotqa`` workspace member (``uv sync --all-packages --group dev``).

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
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from rilixai import Case, OptimizationTargets, optimization_targets_from_prompts, validate_spec

from .agent.prompts import hotpotqa_pydantic_agent_seed_targets
from .config import HotpotQAConfig
from .data.dataset import load_hotpotqa_paper_split
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
    if path is None:
        return hotpotqa_pydantic_agent_seed_targets()
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
    parsed = {str(k): str(v) for k, v in prompts.items()}
    # Guard against a mis-shaped/typo'd file being read as a bare name→text map:
    # ``apply_candidate`` silently ignores unknown component names, so without
    # this a candidate whose keys match nothing would evaluate the *seed*
    # prompts and report that score as the candidate's.
    known = set(hotpotqa_pydantic_agent_seed_targets().to_dict())
    if not (parsed.keys() & known):
        raise ValueError(
            f"Candidate JSON at {path} has no recognized prompt components "
            f"(expected any of {sorted(known)}, got {sorted(parsed)})."
        )
    return optimization_targets_from_prompts(parsed)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _build_spec_for_args(args: argparse.Namespace) -> Any:
    pydantic_agent_model = args.pydantic_agent_model
    if pydantic_agent_model is None:
        # Translate the slash spec (``openai/gpt-4.1-mini``) to PydanticAI's
        # spec (``openai:gpt-4.1-mini``) so callers can share ``--task-model``.
        pydantic_agent_model = args.task_model.replace("/", ":", 1)
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
    validate_spec(spec)
    logger.info(
        "Spec %r validated: %d seed prompt(s) %s.",
        spec.name,
        len(spec.seed_targets.prompts),
        sorted(spec.seed_targets.prompts),
    )
    return 0


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
    summary = {
        "split": args.split,
        "num_cases": report.num_cases,
        "num_errored": report.num_errored,
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
