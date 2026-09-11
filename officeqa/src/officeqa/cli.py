"""``officeqa`` CLI: ``fetch`` (warm the cache), ``run`` and ``evaluate``.

    uv run officeqa fetch    --split test
    uv run officeqa run      --split test --limit 10 --output-dir runs/test-gpt54
    uv run officeqa evaluate --split test --output-dir runs/test-gpt54

``run`` writes ``<output_dir>/results/<uid>.json`` per question plus
``config.json`` and ``summary.json``. ``evaluate`` treats that directory as a
resume point: clean results are reused, anything else is re-run, and
``--rerun`` forces all. All flags default to the baseline configuration.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

from officeqa import config, runner
from officeqa.config import RunConfig
from officeqa.data.corpus import ensure_corpus, fetch_corpus
from officeqa.data.dataset import KNOWN_SPLITS, load_split
from officeqa.data.manifest import build_manifest
from officeqa.evaluation.run_eval import format_summary, select_to_run, summarize
from officeqa.runner import RunResult, read_results, run_split, write_result


logger = logging.getLogger("officeqa")


def _add_run_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--split", choices=KNOWN_SPLITS, default="test")
    p.add_argument(
        "--limit", type=int, default=None, help="first N questions of the split (files are round-robin over strata)"
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--model", default=config.DEFAULT_MODEL, help=f"LiteLLM model string (default {config.DEFAULT_MODEL})"
    )
    p.add_argument("--reasoning-effort", default=config.DEFAULT_REASONING_EFFORT)
    p.add_argument("--tools", default=",".join(config.DEFAULT_TOOLS), help="comma list of fs,repl,web")
    p.add_argument("--corpus", choices=sorted(config.CORPUS_REPRESENTATIONS), default=config.DEFAULT_CORPUS)
    p.add_argument("--search", choices=config.KNOWN_SEARCH_ARMS, default=config.DEFAULT_SEARCH)
    p.add_argument("--n-rollouts", type=int, default=config.DEFAULT_N_ROLLOUTS)
    p.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    p.add_argument("--window-size", type=int, default=config.WINDOW_SIZE)
    p.add_argument("--tool-output-limit", type=int, default=config.TOOL_OUTPUT_LIMIT)
    p.add_argument("--max-retries", type=int, default=config.MAX_RETRIES)
    p.add_argument("--max-concurrent", type=int, default=config.DEFAULT_MAX_CONCURRENT)
    p.add_argument(
        "--task-timeout", type=float, default=config.DEFAULT_TASK_TIMEOUT_S, help="seconds; timeouts score 0"
    )
    p.add_argument(
        "--no-isolate", action="store_true", help="skip the per-question virtualenv (faster; shares site-packages)"
    )


def _cfg_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        tools=tuple(t.strip() for t in args.tools.split(",") if t.strip()),
        corpus=args.corpus,
        search=args.search,
        n_rollouts=args.n_rollouts,
        max_steps=args.max_steps,
        window_size=args.window_size,
        tool_output_limit=args.tool_output_limit,
        max_retries=args.max_retries,
        max_concurrent=args.max_concurrent,
        task_timeout_s=args.task_timeout,
    )


def log_resources(cfg: RunConfig) -> dict[str, object]:
    """Log cores / RAM / requested concurrency; warn when PDF mode oversubscribes cores."""
    cpus = os.cpu_count() or 0
    ram_gb: float | None
    try:
        import psutil

        ram_gb = psutil.virtual_memory().available / 1e9
    except Exception:
        ram_gb = None
    logger.info(
        "cpu_count=%s available_ram=%s max_concurrent=%d corpus=%s",
        cpus,
        f"{ram_gb:.1f} GB" if ram_gb is not None else "unknown",
        cfg.max_concurrent,
        cfg.corpus,
    )
    warned = False
    if cfg.corpus == "pdfs" and cpus and cfg.max_concurrent > cpus:
        warned = True
        logger.warning(
            "--max-concurrent %d exceeds %d cores while --corpus pdfs is active. PDF mode is CPU-bound "
            "(each worker spawns OCR processes); consider --max-concurrent %d.",
            cfg.max_concurrent,
            cpus,
            cpus,
        )
    return {
        "cpu_count": cpus,
        "available_ram_gb": ram_gb,
        "max_concurrent": cfg.max_concurrent,
        "oversubscribed_warning": warned,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _execute(args: argparse.Namespace, *, rerun: bool, resume: bool, summary_only: bool = False) -> int:
    cfg = _cfg_from_args(args)
    out: Path = args.output_dir
    results_dir = out / "results"
    out.mkdir(parents=True, exist_ok=True)

    resources = log_resources(cfg)
    records = load_split(args.split, limit=args.limit)
    corpus = ensure_corpus(cfg.corpus)
    manifest = build_manifest(corpus, default=cfg.corpus)

    existing = read_results(results_dir) if resume else {}
    todo = [] if summary_only else select_to_run(records, existing, rerun=rerun)
    logger.info("split=%s n=%d to_run=%d reused=%d", args.split, len(records), len(todo), len(records) - len(todo))

    if not summary_only:
        _write_json(
            out / "config.json",
            {
                "config": dataclasses.asdict(cfg) | {"tools": list(cfg.tools)},
                "split": args.split,
                "limit": args.limit,
                "uids": [r.uid for r in records],
                "dataset_revision": config.OFFICEQA_DATASET_REVISION,
                "manifest": manifest.to_json(),
                "resources": resources,
                "started_at": time.time(),
                "argv": sys.argv[1:],
            },
        )

    t0 = time.monotonic()
    if todo:

        def on_result(r: RunResult) -> None:
            write_result(r, results_dir)
            logger.info(
                "uid=%s status=%s correct=%d steps=%d tool_calls=%d cost=%s latency=%.0fs",
                r.uid,
                r.status,
                int(r.correct),
                r.steps,
                r.tool_calls,
                f"${r.cost_usd:.2f}" if r.cost_usd is not None else "?",
                r.latency_s,
            )

        run_split(
            todo,
            cfg,
            corpus_root=corpus,
            work_dir=out / "work",
            client_factory=runner.CLIENT_FACTORY,
            on_result=on_result,
            isolate=not args.no_isolate,
        )
    wall = time.monotonic() - t0

    final = read_results(results_dir)
    results = [final[r.uid] for r in records if r.uid in final]
    summary = summarize(results, expected=[r.uid for r in records])
    summary["wall_clock_s"] = wall
    summary["split"] = args.split
    summary["model"] = cfg.model
    summary["corpus"] = cfg.corpus
    summary["tools"] = list(cfg.tools)
    summary["max_concurrent"] = cfg.max_concurrent
    _write_json(out / "summary.json", summary)
    print(format_summary(summary))
    return 0


_GATED_HELP = (
    "Cannot access the gated dataset databricks/officeqa.\n"
    "  1. Request access at https://huggingface.co/datasets/databricks/officeqa\n"
    "  2. Put a read token in HF_TOKEN (see .env.example) or run `hf auth login`"
)


def cmd_fetch(args: argparse.Namespace) -> int:
    reps = tuple(r.strip() for r in args.representations.split(",") if r.strip())
    try:
        root = fetch_corpus(reps)
        manifest = build_manifest(root, default=reps[0] if len(reps) == 1 else config.DEFAULT_CORPUS)
        print(json.dumps(manifest.to_json(), indent=2))
        if args.split:
            records = load_split(args.split)
            print(f"{args.split}: {len(records)} questions loaded at revision {config.OFFICEQA_DATASET_REVISION}")
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        logger.error("%s\n%s", _GATED_HELP, exc)
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "LiteLLM", "primp", "ddgs"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        prog="officeqa", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="download questions + corpus into the cache (no model key needed)")
    f.add_argument(
        "--split", choices=KNOWN_SPLITS, default=None, help="also load this split's questions to verify HF access"
    )
    f.add_argument("--representations", default="pdfs,parsed", help="comma list of pdfs,parsed")
    f.set_defaults(func=cmd_fetch)

    r = sub.add_parser("run", help="run the agent over a split, writing one JSON per question")
    _add_run_flags(r)
    r.add_argument("--rerun", action="store_true", help="ignore existing results in --output-dir")
    r.set_defaults(func=lambda a: _execute(a, rerun=a.rerun, resume=True))

    e = sub.add_parser("evaluate", help="resume an output dir (re-run unclean questions) and print the summary")
    _add_run_flags(e)
    e.add_argument("--rerun", action="store_true", help="force re-running every question")
    e.add_argument("--summary-only", action="store_true", help="aggregate existing results only; never runs the agent")
    e.set_defaults(func=lambda a: _execute(a, rerun=a.rerun, resume=True, summary_only=a.summary_only))

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
