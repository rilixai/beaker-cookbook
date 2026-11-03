"""Prepare real data and evaluate the application without Beaker."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from typesafe_sdk import AsyncTypeSafeClient

from jev_banking77.agent import MODEL, classify, question
from jev_banking77.data import SPLIT_FILES, load_rows, prepare


async def evaluate(args: argparse.Namespace) -> dict[str, float | int]:
    dataset_path = args.data / SPLIT_FILES[args.split]
    rows = load_rows(dataset_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    routing_question = question(args.prompts)
    correct = errors = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncTypeSafeClient(timeout=60) as client:
        with args.output.open("w") as output:
            for row in rows:
                outcome: dict[str, object] = {"id": row.id, "expected": row.expected.intent}
                try:
                    prediction = await classify(row.input.text, client=client, routing_question=routing_question)
                    outcome["prediction"] = prediction.model_dump(mode="json")
                    correct += prediction.intent == row.expected.intent
                except Exception as exc:
                    # Keep errors in the denominator. Do not print provider
                    # response bodies or credentials into benchmark artifacts.
                    outcome["error"] = type(exc).__name__
                    errors += 1
                output.write(json.dumps(outcome) + "\n")
                output.flush()
    metrics: dict[str, float | int] = {"cases": len(rows), "errors": errors, "accuracy": correct / len(rows)}
    args.output.with_suffix(".summary.json").write_text(
        json.dumps({"model": MODEL, "split": args.split, **metrics}, indent=2) + "\n"
    )
    return metrics


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Download and verify the frozen public dataset")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--cache", type=Path, default=Path.home() / ".cache/jev-banking77" / "source-v1")
    run = commands.add_parser("evaluate", help="Call native Jev; uses TYPESAFE_API_KEY and incurs API charges")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    run.add_argument("--prompts", type=Path, default=Path("prompts"))
    run.add_argument("--limit", type=positive_int)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.output, args.cache), indent=2))
    else:
        metrics = asyncio.run(evaluate(args))
        print(json.dumps(metrics, indent=2))
        if metrics["errors"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
