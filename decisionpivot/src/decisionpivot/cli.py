"""Prepare the core track, evaluate the agent without Beaker, and check files for leaked item text."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from decisionpivot import agent
from decisionpivot.data import DEFAULT_CACHE, SPLIT_FILES, Row, fetch_upstream, find_leaks, load_rows, prepare
from decisionpivot.decide import (
    DEFAULT_TARGET,
    PRICE_PER_MILLION_INPUT_TOKENS,
    ChatModel,
    DecisionTarget,
    default_chat_model,
    run_item,
)
from decisionpivot.scoring import latency_and_cost, score_group, upstream_scores, write_run


def target_for(args: argparse.Namespace) -> DecisionTarget:
    """Native Jev, or another decision model behind a TypeSafe-compatible endpoint."""
    if args.model is None and args.base_url is None:
        return DEFAULT_TARGET
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        raise SystemExit(f"Set {args.api_key_env}")
    model = args.model or DEFAULT_TARGET.model
    return DecisionTarget(
        model=model,
        provider=model.partition(":")[0] if ":" in model else DEFAULT_TARGET.provider,
        server_address=(args.base_url or "https://api.typesafe.ai").split("//")[-1].split("/")[0],
        base_url=args.base_url,
        api_key=api_key,
        expected_model=args.expected_model,
    )


async def evaluate(
    args: argparse.Namespace, *, chat_factory: Callable[[], ChatModel] = default_chat_model, client: Any = None
) -> dict[str, Any]:
    rows = load_rows(args.data / SPLIT_FILES[args.split])
    if args.limit is not None:
        rows = rows[: args.limit]
    instructions = agent.load_instructions(args.prompts)
    target = target_for(args)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(row: Row) -> dict[str, Any]:
        records = []
        for item in row.input.items:  # each item alone: the agent never sees the other members
            async with semaphore:
                outcome = await run_item(
                    item,
                    solver=agent.answer,
                    instructions=instructions,
                    target=target,
                    chat_factory=chat_factory,
                    client=client,
                )
            records.append(outcome.record())
        return {"items": records}

    outputs = await asyncio.gather(*(one(row) for row in rows))
    by_row = {row.id: output for row, output in zip(rows, outputs, strict=True)}
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for row in rows:
        scored = score_group(row, by_row[row.id])
        cases.append(
            {
                "id": row.id,
                "family": row.input.family,
                "level": row.expected.level,
                "objective": scored.objective,
                "fields": scored.field_scores,
                "items": [vars(s) for s in scored.items],
            }
        )
    (args.output / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    (args.output / "outputs.jsonl").write_text("".join(json.dumps({"id": r.id, **by_row[r.id]}) + "\n" for r in rows))
    upstream = fetch_upstream(args.cache)
    write_run(args.output, rows, by_row, upstream)
    families = sorted({row.input.family for row in rows})
    scores = upstream_scores(args.output, upstream, families, resamples=args.resamples)
    item_outputs = [o for r in rows for o in by_row[r.id]["items"]]
    price = PRICE_PER_MILLION_INPUT_TOKENS if target.model == DEFAULT_TARGET.model else None
    errors = [o for o in item_outputs if o["error"] is not None]
    summary: dict[str, Any] = {
        "model": target.model,
        "split": args.split,
        "prompts": str(args.prompts),
        "groups": len(rows),
        "items": len(item_outputs),
        "failed_items": len(errors),
        "failed_by_status": _count(str(o["error"]).split(":")[0] for o in errors),
        "upstream_core": scores["core"],
        "upstream_per_family": {family: _family_row(r) for family, r in sorted(scores["tasks"].items())},
        **latency_and_cost(item_outputs, price_per_million_input_tokens=price),
        "upstream_report": scores["report"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _count(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def _family_row(r: dict[str, Any]) -> dict[str, Any]:
    keys = ("accuracy", "pair_accuracy", "pair_direction", "auto_at_5", "acc_at_090", "share_at_090", "ece")
    return {
        **{k: r.get(k) for k in keys},
        "level3_pair_accuracy": (r.get("levels") or {}).get("3", {}).get("pair_accuracy"),
        "items": r.get("rows"),
        "coverage": r.get("coverage"),
    }


def leak_check(data: Path, paths: list[Path]) -> list[dict[str, str]]:
    rows = [row for name in SPLIT_FILES.values() for row in load_rows(data / name)]
    files: dict[str, str] = {}
    for path in paths:
        for file in sorted(path.rglob("*") if path.is_dir() else [path]):
            if file.is_file():
                files[str(file)] = file.read_text(encoding="utf-8", errors="replace")
    if not files:
        raise SystemExit(f"No files to check under {[str(p) for p in paths]}")
    return find_leaks(rows, files)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Regenerate the core track from the pinned generators and split it")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    prep.add_argument(
        "--groups-per-family", type=positive_int, help="keep the first N groups of each family per split (small runs)"
    )
    run = commands.add_parser("evaluate", help="Run the agent; uses TYPESAFE_API_KEY and incurs API charges")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--split", choices=tuple(SPLIT_FILES), default="validation")
    run.add_argument("--prompts", type=Path, default=Path("prompts"))
    run.add_argument("--limit", type=positive_int, help="first N contrast groups of the split")
    run.add_argument("--concurrency", type=positive_int, default=16)
    run.add_argument("--output", type=Path, required=True, help="output directory")
    run.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    run.add_argument("--resamples", type=int, default=1000, help="bootstrap resamples for upstream intervals")
    run.add_argument("--model", help="decision model to compare; default: the pinned Jev model")
    run.add_argument("--base-url", help="TypeSafe-compatible decision endpoint for --model")
    run.add_argument("--api-key-env", help="environment variable holding the key for --base-url")
    run.add_argument("--expected-model", help="fail answers that report another model version")
    leak = commands.add_parser("leak-check", help="Fail if item text appears in the given files")
    leak.add_argument("--data", type=Path, required=True)
    leak.add_argument("--path", type=Path, action="append", help="file or directory (repeatable)")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.output, args.cache, groups_per_family=args.groups_per_family), indent=2))
        return
    if args.command == "leak-check":
        paths = args.path or [Path("prompts"), Path("src/decisionpivot/agent.py")]
        leaks = leak_check(args.data, paths)
        for leak_ in leaks[:20]:
            print(json.dumps(leak_))
        print(json.dumps({"checked": [str(p) for p in paths], "leaks": len(leaks)}))
        if leaks:
            raise SystemExit(1)
        return
    summary = asyncio.run(evaluate(args))
    print(json.dumps({k: v for k, v in summary.items() if k != "upstream_per_family"}, indent=2))
    if summary["failed_items"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
