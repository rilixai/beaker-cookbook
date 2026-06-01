"""CLI entrypoint for **local** HotpotQA agent runs.

Run as ``python -m hotpotqa.cli ...`` after installing the cookbook's
``hotpotqa`` workspace member (``uv sync --all-packages --group dev``).

For **remote** runs on the rilixai Modal sandbox path, see
``hotpotqa/sandbox.py``. A future PR will collapse both entry points
into one subcommand-style CLI.

With no flags, ``evaluate`` runs the seed candidate on the GEPA paper's
exact 300-case test slice under the paper's fullwiki / k=7 setup. Data
loading is bit-faithful to the artifact: source is the HotpotQA *train*
split (the 90k-case pool); we slice ``[0, 40%)`` for test,
``[40%, 80%)`` for val, ``[80%, 100%)`` for train, then sample with
``random.Random(1)``. The result is the same 300/300/150 picks the GEPA
paper reports against, so absolute numbers compare apples-to-apples.

``optimize`` defaults are also paper-aligned: ``--max-metric-calls 6871``
matches the artifact's HotpotQA budget; the optimization objective is
pure exact match (the paper's ``frac=1.0`` setup). For full paper parity
pass ``--reflection-model openai/gpt-4.1`` — the paper uses GPT-4.1 as
the reflection LM (stronger than the GPT-4.1-mini task LM); we leave it
unset by default so the task LM is reused (cheaper, but a weaker
reflection signal).

Example paper-faithful run::

    uv run python -m hotpotqa.cli optimize \\
        --reflection-model openai/gpt-4.1 \\
        --output-dir hotpotqa_results/optimize

    uv run python -m hotpotqa.cli evaluate \\
        --candidate-json hotpotqa_results/optimize/best_candidate.json \\
        --output-dir hotpotqa_results/after
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from rilixai.prompt_optimization.evaluation import (
    evaluate_candidate_on_cases,
    field_accuracy_rows,
    serialize_eval_outputs,
)
from rilixai.prompt_optimization.models import Sample, PromptCandidate
from rilixai.prompt_optimization.optimization import extract_best_candidate, summarize_gepa_result_metadata
from rilixai.prompt_optimization.spec import (
    PromptOptimizationRunConfig,
    build_adapter_from_spec,
    run_optimization_from_spec,
)

from .agent.prompts import hotpotqa_pydantic_agent_seed_candidate
from .config import HotpotQAConfig
from .data.dataset import load_hotpotqa_paper_split
from .optimization.spec import build_hotpotqa_spec


logger = logging.getLogger("hotpotqa")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "HotpotQA benchmark for the rilixai prompt optimizer. With no "
            "flags, ``evaluate`` runs on the GEPA paper's held-out test "
            "split of 300 cases under the paper's fullwiki / k=7 setup."
        ),
    )
    parser.add_argument(
        "command",
        choices=("optimize", "evaluate"),
        help="`optimize` runs GEPA; `evaluate` scores a single candidate.",
    )
    # Split sizes mirror the GEPA paper's HotpotQA setup: 150 train / 300 val / 300 test.
    # These cap how many cases get sampled from each artifact-fraction slice
    # (test = first 40% of HotpotQA train, val = next 40%, train = last 20%).
    parser.add_argument(
        "--train-size",
        type=int,
        default=150,
        help=(
            "How many train cases to draw from the paper's [80%%, 100%%) slice of "
            "HotpotQA train. Default 150 matches the artifact's `trim_dataset` size."
        ),
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=300,
        help=(
            "How many validation cases to draw from the paper's [40%%, 80%%) slice of "
            "HotpotQA train. Used by GEPA's internal Pareto selection during "
            "``optimize`` and by ``evaluate --split validation``. Default 300."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=300,
        help=(
            "How many test cases to draw from the paper's [0%%, 40%%) slice of "
            "HotpotQA train. This is the held-out set ``evaluate`` reads by default — "
            "the paper-faithful number to report. Default 300."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("test", "validation"),
        default="test",
        help=(
            "``evaluate`` only: which split to score the candidate on. ``test`` "
            "(default) is the held-out paper-faithful number. ``validation`` is "
            "the set GEPA's Pareto selection used during ``optimize`` and is "
            "mainly useful for sanity-checking optimize runs (numbers there are "
            "Pareto-biased and shouldn't be reported)."
        ),
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=6871,
        help=(
            "GEPA metric-call budget for `optimize`. Default 6871 matches the "
            "artifact's known_max_calls for ('HotpotQABench', 'HotpotMultiHop', "
            "'MIPROv2-Heavy') — the budget the paper's GEPA HotpotQA runs use "
            "for fair comparison vs MIPROv2-Heavy. GEPA typically completes "
            "well under budget (~1500 calls on GPT-4.1-mini per the artifact "
            "figures). Drop to ~300 for cheap smoke runs."
        ),
    )
    parser.add_argument("--reflection-minibatch-size", type=int, default=3)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Seed for the optimizer (`optimize` only). Note this does NOT control "
            "data sampling — case selection is pinned to the artifact's `random.Random(1)` "
            "so the 300/300/150 picks bit-match the paper across runs."
        ),
    )
    parser.add_argument(
        "--retrieval",
        choices=("fullwiki", "distractor"),
        default="fullwiki",
        help=(
            "Retrieval corpus. ``fullwiki`` (default, paper parity) uses bm25s "
            "over the 2017 Wikipedia abstracts dump (~5M docs, downloaded and "
            "indexed lazily). ``distractor`` uses BM25 over the 10 "
            "per-case distractor paragraphs HotpotQA ships — fast and "
            "test-friendly but easier than the paper's setup."
        ),
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
        help=(
            "PydanticAI model spec for the outer agent (e.g. "
            "'openai:gpt-4.1-mini'). If omitted, ``--task-model`` is reused "
            "with ``openai/...`` rewritten to ``openai:...``."
        ),
    )
    parser.add_argument(
        "--task-model",
        type=str,
        default="openai/gpt-4.1-mini",
        help=(
            "Default model used by the PydanticAI outer agent and the raw "
            "OpenAI call inside the summarize tool. Slash form like "
            "``openai/gpt-4.1-mini`` is auto-translated to PydanticAI's "
            "``openai:gpt-4.1-mini`` spec."
        ),
    )
    parser.add_argument(
        "--task-temperature",
        type=float,
        default=0.0,
        help=(
            "Sampling temperature for the task LLM. Defaults to ``0.0`` so "
            "baseline + post-optimize evaluations are reproducible. Applied "
            "to the PydanticAI outer Agent's model settings and the raw "
            "OpenAI call inside the summarize tool."
        ),
    )
    parser.add_argument(
        "--reflection-model",
        type=str,
        default=None,
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
        default=Path("hotpotqa_run"),
        help="Directory where results and reflection artifacts are written.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="HuggingFace datasets cache directory (defaults to the user cache).",
    )
    return parser.parse_args(argv)


def _load_splits_for_command(args: argparse.Namespace) -> dict[str, list[Sample]]:
    """Load only the splits the requested command + split flag actually need.

    Paper-faithful by default: mirrors the GEPA artifact's data pipeline
    bit-for-bit. The HotpotQA *train* split is the source; we slice it
    ``[0, 40%)`` → test, ``[40%, 80%)`` → validation, ``[80%, 100%)`` →
    train, then ``random.Random(1).sample(slice, size)`` to draw the
    artifact's 300/300/150. The HF dataset config tracks ``--retrieval``
    (``fullwiki`` config when retrieval is fullwiki — matching the paper;
    ``distractor`` config when retrieval is distractor — needed for the
    per-case 10-paragraph context). See
    :func:`hotpotqa.data.dataset.load_hotpotqa_paper_split`
    for the full provenance.
    """
    cache_dir = str(args.cache_dir) if args.cache_dir else None
    config = "fullwiki" if args.retrieval == "fullwiki" else "distractor"
    splits: dict[str, list[Sample]] = {}

    if args.command == "optimize":
        if args.train_size > 0:
            splits["train"] = load_hotpotqa_paper_split(
                "train",
                max_cases=args.train_size,
                config=config,
                cache_dir=cache_dir,
            )
        if args.val_size <= 0:
            raise ValueError("optimize requires --val-size > 0.")
        splits["validation"] = load_hotpotqa_paper_split(
            "validation",
            max_cases=args.val_size,
            config=config,
            cache_dir=cache_dir,
        )
        return splits

    # evaluate
    if args.split == "test":
        if args.test_size <= 0:
            raise ValueError("evaluate --split test requires --test-size > 0.")
        splits["test"] = load_hotpotqa_paper_split(
            "test",
            max_cases=args.test_size,
            config=config,
            cache_dir=cache_dir,
        )
    else:
        if args.val_size <= 0:
            raise ValueError("evaluate --split validation requires --val-size > 0.")
        splits["validation"] = load_hotpotqa_paper_split(
            "validation",
            max_cases=args.val_size,
            config=config,
            cache_dir=cache_dir,
        )
    return splits


def _load_candidate(path: Path | None) -> PromptCandidate:
    """Resolve the candidate to evaluate, defaulting to the agent seed."""
    if path is None:
        return hotpotqa_pydantic_agent_seed_candidate()
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

    The adapter calls the runtime once per case (potentially concurrently
    up to ``--max-concurrency``). We count completions under a lock,
    track wall time from the first start, and emit progress lines.

    Throttling: a progress line fires when *any* of these trigger —
    (a) it's the first or last completion, (b) at least ``log_every_n``
    cases have completed since the last line, (c) at least
    ``log_every_seconds`` have elapsed since the last line. This keeps
    long ``optimize`` runs (up to ``max_metric_calls`` completions) from
    spamming thousands of INFO lines while still showing forward motion
    when nothing has happened in a while.

    For ``evaluate``, ``total`` is the held-out split size, so each
    line shows percent + ETA. For ``optimize``, pass the GEPA
    metric-call budget as ``total`` to get the same percent + ETA
    against the cap; GEPA usually finishes well under the cap so the
    final line will show < 100%.
    """
    state: dict[str, Any] = {
        "done": 0,
        "started": None,
        "last_log_count": 0,
        "last_log_time": 0.0,
    }
    # ``asyncio.Lock`` (vs ``threading.Lock``) is the idiomatic choice
    # since this wrapper runs entirely on the event loop and the rilixai
    # adapter dispatches concurrent cases as ``asyncio.gather`` tasks.
    # Critical sections still contain no ``await`` today, but switching
    # to ``asyncio.Lock`` means a future ``await`` slipped into one of
    # them would yield to the loop instead of blocking the whole thread.
    lock = asyncio.Lock()

    async def _wrapped(**kwargs: Any) -> Any:
        async with lock:
            if state["started"] is None:
                state["started"] = time.monotonic()
        try:
            result = await runtime(**kwargs)
        finally:
            # CRITICAL: do NOT `return` from this `finally` block. In
            # Python, a `return` inside `finally` overrides the try
            # block's return value — silently swallowing the runtime's
            # actual result and handing the caller `None` instead.
            # Earlier versions of this wrapper had `if not should_log:
            # return` here, which dropped ~95%% of optimize-mode case
            # results to None and corrupted GEPA's optimization signal.
            # Always update progress state and emit the log line
            # conditionally, then fall out of the `finally` so the
            # outer `return result` (or re-raised exception) takes effect.
            async with lock:
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


def _summarize_agent_tool_usage(outputs: Sequence[Any]) -> dict[str, Any] | None:
    """Aggregate per-case agent tool-call counts into evaluate-summary stats.

    Reads ``run_metrics["tool_counts"]`` (the canonical top-level dict
    rilixai's adapter consumes — keys are ``hotpotqa_``-prefixed for
    namespace safety against other benchmarks' counts) and
    ``run_metrics["hotpotqa"]["num_total_steps"]`` from every output.
    The ``hotpotqa_`` prefix is stripped for display so users see the
    raw tool names (``retrieve_k``, ``summarize``, ``finish``).
    Returns ``None`` if no case carries the expected agent metadata.
    """
    prefix = "hotpotqa_"
    per_tool_totals: dict[str, int] = {}
    total_calls_sum = 0
    cases_seen = 0
    for output in outputs:
        run_metrics = getattr(output, "run_metrics", None) or {}
        if not isinstance(run_metrics, Mapping):
            continue
        hotpotqa = run_metrics.get("hotpotqa")
        if not isinstance(hotpotqa, Mapping):
            continue
        tool_counts = run_metrics.get("tool_counts")
        if not isinstance(tool_counts, Mapping):
            continue
        cases_seen += 1
        total_calls_sum += int(hotpotqa.get("num_total_steps", 0) or 0)
        for raw_name, count in tool_counts.items():
            if not isinstance(raw_name, str):
                continue
            display_name = raw_name[len(prefix) :] if raw_name.startswith(prefix) else raw_name
            try:
                per_tool_totals[display_name] = per_tool_totals.get(display_name, 0) + int(count)
            except (TypeError, ValueError):
                continue
    if cases_seen == 0:
        return None
    return {
        "num_cases": cases_seen,
        "avg_total_calls": total_calls_sum / cases_seen,
        "avg_calls_per_tool": {tool: total / cases_seen for tool, total in sorted(per_tool_totals.items())},
    }


def _format_tool_usage_table(usage: Mapping[str, Any]) -> str:
    lines = [f"  avg_total_calls = {float(usage['avg_total_calls']):.2f}  (over {usage['num_cases']} cases)"]
    per_tool = usage.get("avg_calls_per_tool", {})
    if isinstance(per_tool, Mapping):
        for tool_name, avg in per_tool.items():
            lines.append(f"  {tool_name:<24s} avg_calls_per_case={float(avg):.2f}")
    return "\n".join(lines)


def _placeholder_cases_for_spec(splits: dict[str, list[Sample]]) -> dict[str, Sequence[Sample]]:
    """Build a ``samples_by_split`` map the rilixai spec is happy to validate.

    ``PromptOptimizationSpec`` validates that every split it sees is a
    non-empty sequence of ``Sample``. For ``evaluate`` runs we only need
    the split being scored, but the spec still needs a ``samples_by_split``
    mapping. Pass the loaded splits through directly; for optimize runs
    we additionally need ``train``+``validation`` which ``_load_splits_for_command``
    already produced.
    """
    return {name: list(cases) for name, cases in splits.items()}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    splits = _load_splits_for_command(args)
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
    spec = build_hotpotqa_spec(
        samples_by_split=_placeholder_cases_for_spec(splits),
        model=args.task_model,
        max_concurrency=args.max_concurrency,
        config=config,
    )

    # Wrap the spec's runtime with progress logging. For ``evaluate`` the
    # total is the split size — one line per case is fine (300 lines max).
    # For ``optimize`` the total is the metric-call budget (GEPA usually
    # finishes well under), and we throttle to one line per 25 completions
    # or per 30 seconds (whichever fires first) — otherwise 2000-call runs
    # would emit 2000 progress lines.
    if args.command == "evaluate":
        progress_total: int | None = len(splits.get(args.split, []))
        progress_label = f"evaluate:{args.split}"
        progress_every_n = 1
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
        run_config = PromptOptimizationRunConfig.from_spec(
            spec,
            max_metric_calls=args.max_metric_calls,
            reflection_minibatch_size=args.reflection_minibatch_size,
            reflection_model=args.reflection_model,
            run_dir=str(args.output_dir),
            reflection_artifact_dir=str(args.output_dir / "reflection_artifacts"),
            seed=args.seed,
        )
        optimize_started = time.monotonic()
        logger.info("Starting optimize (max_metric_calls=%d)...", args.max_metric_calls)
        result = run_optimization_from_spec(spec, run_config)
        logger.info("optimize complete in %s", _fmt_hms(time.monotonic() - optimize_started))
        best = extract_best_candidate(result)
        metadata = summarize_gepa_result_metadata(result)
        _write_json(args.output_dir / "best_candidate.json", best.to_dict())
        _write_json(args.output_dir / "gepa_metadata.json", metadata)
        logger.info("Best candidate written to %s", args.output_dir / "best_candidate.json")
        logger.info("GEPA metadata: %s", metadata)
        return 0

    # evaluate
    adapter = build_adapter_from_spec(spec)
    candidate = _load_candidate(args.candidate_json)
    target_cases = list(splits.get(args.split, []))
    if not target_cases:
        logger.error("evaluate command got no cases for --split %s.", args.split)
        return 2
    eval_started = time.monotonic()
    logger.info("Starting evaluate on split=%s (%d cases)...", args.split, len(target_cases))
    report = evaluate_candidate_on_cases(adapter=adapter, candidate=candidate, samples=target_cases)
    eval_elapsed = time.monotonic() - eval_started
    logger.info("evaluate complete in %s (%d cases)", _fmt_hms(eval_elapsed), len(target_cases))
    summary: dict[str, Any] = {
        "split": args.split,
        "num_cases": len(target_cases),
        "weighted_objective": report.weighted_objective,
        "field_accuracies": report.field_accuracies,
        "field_sample_counts": report.field_sample_counts,
    }
    tool_usage = _summarize_agent_tool_usage(report.outputs)
    if tool_usage is not None:
        summary["agent_tool_usage"] = tool_usage
    _write_json(args.output_dir / "eval_summary.json", summary)
    _write_json(args.output_dir / "eval_outputs.json", serialize_eval_outputs(report.outputs))
    logger.info(
        "Split=%s | weighted_objective=%.4f over %d cases",
        args.split,
        report.weighted_objective,
        len(target_cases),
    )
    logger.info("Field accuracies:\n%s", _format_field_table(report.field_accuracies, report.field_sample_counts))
    if tool_usage is not None:
        logger.info("Agent tool usage:\n%s", _format_tool_usage_table(tool_usage))
    return 0


if __name__ == "__main__":
    sys.exit(main())
