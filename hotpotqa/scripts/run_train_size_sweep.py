"""Sweep HotpotQA agent post-optimize EM across train-size points.

For each train size in ``--train-sizes``, this script:

1. Runs ``hotpotqa.cli optimize`` with that ``--train-size``.
2. Runs ``hotpotqa.cli evaluate --split test`` against the optimized
   ``best_candidate.json`` from step 1.
3. Appends one row per train_size to a consolidated
   ``sweep_summary.csv`` + ``sweep_summary.json`` in the output root.

Resumable: if ``<output_root>/train<N>/after/eval_summary.json`` already
exists when ``--skip-existing`` is passed, that point is skipped. Useful
when one point crashed mid-sweep or when you want to extend an existing
sweep with new train-size points.

Holds every other knob constant so the only varying axis is
``--train-size``:
    * ``--task-model openai/gpt-4.1-mini-2025-04-14``
    * ``--task-temperature 0.0``
    * ``--reflection-model openai/gpt-4.1``
    * ``--val-size 300``
    * ``--test-size 300``
    * ``--seed 0``

The GEPA metric-call budget is configurable via ``--max-metric-calls``
(default 2000 — above GEPA's median actual usage of ~1,550 on
gpt-4.1-mini HotpotQA per the artifact's figures, with headroom and
~70% savings vs the paper's 6871 cap). Pass ``--max-metric-calls 6871``
for a paper-exact run.

The CSV is written after every train_size point finishes so you can
plot mid-sweep without losing progress on crashes.

Example::

    uv run python -m hotpotqa.scripts.run_train_size_sweep \\
        --output-root hotpotqa_sweep \\
        --skip-existing
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from hotpotqa.cli import _fmt_hms


logger = logging.getLogger("hotpotqa.sweep")


DEFAULT_TRAIN_SIZES = (25, 50, 100, 150, 300)

# Default GEPA metric-call budget. The artifact's `known_max_calls` for
# HotpotQA is 6871 — but that cap exists for fair comparison vs
# MIPROv2-Heavy, *not* because GEPA needs it. The artifact's figure data
# shows GEPA's actual usage is ~1,550 calls on average for gpt-4.1-mini.
# 2000 sits above the median with headroom while saving ~70% of compute
# vs the paper cap. Override with ``--max-metric-calls`` for a stricter
# paper-faithful run.
DEFAULT_MAX_METRIC_CALLS = 2000

# Knobs we hold constant across every sweep run so the only varying
# axis is ``--train-size``. Sourced from the calibrated paper-faithful
# settings established in earlier commits.
_STATIC_CLI_ARGS = (
    "--task-model",
    "openai/gpt-4.1-mini-2025-04-14",
    "--task-temperature",
    "0.0",
    "--reflection-model",
    "openai/gpt-4.1",
    "--val-size",
    "300",
    "--test-size",
    "300",
    "--seed",
    "0",
)


def _shared_cli_args(max_metric_calls: int) -> list[str]:
    """Return the static knobs plus the (sweep-configurable) metric-call budget."""
    return ["--max-metric-calls", str(max_metric_calls), *_STATIC_CLI_ARGS]


async def _run_cli(args: list[str], log_label: str) -> int:
    """Spawn the CLI as a subprocess and stream its logs straight through.

    Returns the process exit code. We don't capture stdout/stderr —
    leaving them attached to the parent so the per-case progress lines
    the CLI emits are visible live in the sweep terminal.
    """
    full = [sys.executable, "-m", "hotpotqa.cli", *args]
    logger.info("[%s] starting: %s", log_label, " ".join(full[2:]))
    proc = await asyncio.create_subprocess_exec(*full)
    code = await proc.wait()
    logger.info("[%s] exit code %s", log_label, code)
    return code


async def _run_train_size(
    *,
    train_size: int,
    output_root: Path,
    candidate_filename: str,
    skip_existing: bool,
    max_metric_calls: int,
) -> dict[str, Any] | None:
    """Optimize then evaluate one train_size point.

    Returns a flat summary dict ready for CSV writing, or ``None`` on
    any unrecoverable failure.
    """
    label = f"train{train_size}"
    train_dir = output_root / f"train{train_size}"
    optimize_dir = train_dir / "optimize"
    after_dir = train_dir / "after"
    candidate_path = optimize_dir / candidate_filename
    eval_summary_path = after_dir / "eval_summary.json"
    shared_args = _shared_cli_args(max_metric_calls)

    if skip_existing and eval_summary_path.exists():
        logger.info("[%s] eval_summary already exists — loading + skipping run", label)
        return _load_summary(eval_summary_path, train_size=train_size, optimize_dir=optimize_dir)

    optimize_seconds: float | None = None
    if not candidate_path.exists():
        opt_args = [
            "optimize",
            "--train-size",
            str(train_size),
            "--output-dir",
            str(optimize_dir),
            *shared_args,
        ]
        started = time.monotonic()
        try:
            code = await _run_cli(opt_args, f"{label}/optimize")
        except Exception:
            logger.exception("[%s] optimize raised", label)
            return None
        if code != 0 or not candidate_path.exists():
            logger.error("[%s] optimize failed (exit=%s, candidate_exists=%s)", label, code, candidate_path.exists())
            return None
        optimize_seconds = time.monotonic() - started
    else:
        logger.info("[%s] best_candidate.json already exists — skipping optimize", label)

    eval_args = [
        "evaluate",
        "--split",
        "test",
        "--candidate-json",
        str(candidate_path),
        "--output-dir",
        str(after_dir),
        *shared_args,
    ]
    try:
        code = await _run_cli(eval_args, f"{label}/evaluate")
    except Exception:
        logger.exception("[%s] evaluate raised", label)
        return None
    if code != 0 or not eval_summary_path.exists():
        logger.error("[%s] evaluate failed (exit=%s)", label, code)
        return None
    summary = _load_summary(eval_summary_path, train_size=train_size, optimize_dir=optimize_dir)
    if summary is not None and optimize_seconds is not None:
        summary["optimize_wall_seconds"] = round(optimize_seconds, 1)
    return summary


def _load_summary(
    eval_summary_path: Path,
    *,
    train_size: int,
    optimize_dir: Path,
) -> dict[str, Any] | None:
    """Flatten the per-point eval_summary.json + optional GEPA metadata into one row."""
    try:
        raw = json.loads(eval_summary_path.read_text())
    except Exception:
        logger.exception("Failed to parse %s", eval_summary_path)
        return None
    field_accs = raw.get("field_accuracies", {}) or {}
    summary: dict[str, Any] = {
        "train_size": train_size,
        "num_cases": raw.get("num_cases"),
        "weighted_objective": raw.get("weighted_objective"),
        "em": field_accs.get("answer"),
        "f1": field_accs.get("answer_f1"),
        "supporting_titles_recall": field_accs.get("supporting_titles_recall"),
    }

    # GEPA metadata — the keys here MUST match what
    # ``rilixai.prompt_optimization.optimization.summarize_gepa_result_metadata``
    # actually writes (it's the producer of ``gepa_metadata.json`` via
    # the CLI). Pulling unrelated keys silently drops every metadata
    # field except ``best_val_score`` from the sweep CSV.
    meta_path = optimize_dir / "gepa_metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            for key in (
                "best_val_score",
                "best_idx",
                "total_metric_calls",
                "num_full_val_evals",
                "num_candidates",
                "seed",
            ):
                if key in meta:
                    summary[key] = meta[key]
        except Exception:
            logger.exception("Failed to parse %s", meta_path)

    # Agent tool-usage stats (avg per-tool call counts) — populated by the
    # CLI's ``_summarize_agent_tool_usage``.
    agent_usage = raw.get("agent_tool_usage")
    if isinstance(agent_usage, dict):
        summary["avg_total_tool_calls"] = agent_usage.get("avg_total_calls")
        per_tool = agent_usage.get("avg_calls_per_tool")
        if isinstance(per_tool, dict):
            for tool_name, avg in per_tool.items():
                summary[f"avg_calls_{tool_name}"] = avg

    return summary


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write rows to CSV with a stable preferred column order."""
    if not rows:
        path.write_text("")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    preferred = [
        "train_size",
        "em",
        "f1",
        "supporting_titles_recall",
        "num_cases",
        "weighted_objective",
        "best_val_score",
        "best_idx",
        "total_metric_calls",
        "num_full_val_evals",
        "num_candidates",
        "seed",
        "avg_total_tool_calls",
        "optimize_wall_seconds",
    ]
    ordered = [k for k in preferred if k in keys] + [k for k in keys if k not in preferred]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


async def _main_async(args: argparse.Namespace) -> int:
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    total_points = len(args.train_sizes)
    sweep_started = time.monotonic()
    logger.info("=" * 70)
    logger.info(
        "[SWEEP] starting: %d train-size points, budget=%d/run",
        total_points,
        args.max_metric_calls,
    )
    logger.info("[SWEEP] points: %s", args.train_sizes)
    logger.info("=" * 70)
    for point_index, train_size in enumerate(args.train_sizes, start=1):
        logger.info("=" * 70)
        logger.info("[SWEEP] point %d/%d — train_size=%d", point_index, total_points, train_size)
        logger.info("=" * 70)
        point_started = time.monotonic()
        row = await _run_train_size(
            train_size=train_size,
            output_root=output_root,
            candidate_filename=args.candidate_filename,
            skip_existing=args.skip_existing,
            max_metric_calls=args.max_metric_calls,
        )
        if row is not None:
            all_rows.append(row)
        _write_csv(all_rows, output_root / "sweep_summary.csv")
        (output_root / "sweep_summary.json").write_text(json.dumps(all_rows, indent=2, default=str))
        point_elapsed = time.monotonic() - point_started
        sweep_elapsed = time.monotonic() - sweep_started
        avg_per_point = sweep_elapsed / point_index
        remaining_points = total_points - point_index
        eta = avg_per_point * remaining_points
        em_str = f"{float(row['em']) * 100:.2f}%" if row is not None and row.get("em") is not None else "(no score)"
        logger.info(
            "[SWEEP] point %d/%d complete (train_size=%d) in %s | EM: %s | sweep elapsed=%s eta=%s | rows=%d",
            point_index,
            total_points,
            train_size,
            _fmt_hms(point_elapsed),
            em_str,
            _fmt_hms(sweep_elapsed),
            _fmt_hms(eta),
            len(all_rows),
        )
    logger.info("=" * 70)
    logger.info(
        "[SWEEP] complete: %d/%d points in %s. CSV: %s",
        len(all_rows),
        total_points,
        _fmt_hms(time.monotonic() - sweep_started),
        output_root / "sweep_summary.csv",
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_TRAIN_SIZES),
        help="Train sizes to sweep. Default: 25 50 100 150 300.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("hotpotqa_sweep"),
        help="Where to write per-point subdirs + sweep_summary.csv.",
    )
    parser.add_argument(
        "--candidate-filename",
        type=str,
        default="best_candidate.json",
        help="Filename CLI optimize writes inside its --output-dir.",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=DEFAULT_MAX_METRIC_CALLS,
        help=(
            "GEPA metric-call budget per train_size point. Defaults to "
            f"{DEFAULT_MAX_METRIC_CALLS} — above GEPA's median actual usage on "
            "gpt-4.1-mini HotpotQA (~1,550 per the artifact's figures) with "
            "headroom, while saving ~70%% vs the paper's 6871 cap. Bump to 6871 "
            "for the paper's exact budget."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse points whose eval_summary.json already exists. Makes the sweep resumable.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
