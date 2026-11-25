"""Prepare the pinned data and evaluate the agent without Beaker."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mmdm import agent
from mmdm.data import SPLIT_FILES, Row, load_rows, parse_categories, prepare
from mmdm.decide import (
    DEFAULT_TARGET,
    PRICE_PER_MILLION_INPUT_TOKENS,
    ChatModel,
    DecisionTarget,
    default_chat_model,
    run_question,
)
from mmdm.scoring import score, summarize, upstream_cross_check, upstream_prediction, upstream_row


def case_record(row: Row, output: dict[str, Any]) -> dict[str, Any]:
    scored = score(row.id, row.input, row.expected, output)
    return {
        "id": row.id,
        "category": row.expected.task_category,
        "kind": scored.kind,
        "task": row.input.task,
        "option_count": len(row.input.options),
        "objective": scored.objective,
        "fields": scored.field_scores,
        "expected_top": scored.expected_top,
        "predicted_top": scored.predicted_top,
        "error": scored.error,
        "latency_s": output["latency_s"],
        "decision_calls": output["decision_calls"],
        "decision_input_tokens": output["decision_input_tokens"],
        "decision_output_tokens": output["decision_output_tokens"],
        "models": output["models"],
        "prediction": output["probabilities"],
    }


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
        async with semaphore:
            outcome = await run_question(
                row.input,
                solver=agent.answer,
                instructions=instructions,
                target=target,
                chat_factory=chat_factory,
                client=client,
            )
        return outcome.record()

    outputs = await asyncio.gather(*(one(row) for row in rows))
    records = [case_record(row, output) for row, output in zip(rows, outputs, strict=True)]
    price = PRICE_PER_MILLION_INPUT_TOKENS if target.model == DEFAULT_TARGET.model else None
    summary: dict[str, Any] = {
        "model": target.model,
        "split": args.split,
        "prompts": str(args.prompts),
        **summarize(records, price_per_million_input_tokens=price),
    }
    predictions = [upstream_prediction(r.id, r.input, o) for r, o in zip(rows, outputs, strict=True)]
    summary["upstream_check"] = upstream_cross_check(
        [upstream_row(r.id, r.input, r.expected) for r in rows], predictions, summary
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    upstream_path = args.output.with_suffix(".upstream-predictions.jsonl")
    upstream_path.write_text("".join(json.dumps(p) + "\n" for p in predictions))
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Download, verify and split the pinned MMDM release")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--categories", help="comma-separated task categories; default: all 16")
    prep.add_argument("--cache", type=Path, default=Path.home() / ".cache/mmdm/source-v1")
    run = commands.add_parser("evaluate", help="Run the agent; uses TYPESAFE_API_KEY and incurs API charges")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--split", choices=tuple(SPLIT_FILES), default="validation")
    run.add_argument("--prompts", type=Path, default=Path("prompts"))
    run.add_argument("--limit", type=positive_int)
    run.add_argument("--concurrency", type=positive_int, default=8)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--model", help="decision model to compare; default: the pinned Jev model")
    run.add_argument("--base-url", help="TypeSafe-compatible decision endpoint for --model")
    run.add_argument("--api-key-env", help="environment variable holding the key for --base-url")
    run.add_argument("--expected-model", help="fail answers that report another model version")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.output, args.cache, parse_categories(args.categories)), indent=2))
        return
    summary = asyncio.run(evaluate(args))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("categories", "upstream_check")}, indent=2))
    print(json.dumps({"upstream_check_matches": summary["upstream_check"]["matches"]}))
    failed = summary["coverage"]["failed"]
    if failed or not summary["upstream_check"]["matches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
