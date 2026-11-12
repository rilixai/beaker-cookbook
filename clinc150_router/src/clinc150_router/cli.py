"""Prepare real data and evaluate the application without Beaker."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

from openai import AsyncOpenAI

from clinc150_router.agent import OUT_OF_SCOPE, ChatModel, Prediction, route
from clinc150_router.data import SPLIT_FILES, Row, load_rows, prepare


# Same model and effort the Beaker integration pins for runs without a model choice.
MODEL = "gpt-5.6-luna"
REASONING_EFFORT: Final = "low"


class _NoSpan:
    def output(self, value: Any, /) -> None:
        del value

    def usage(self, *, input_tokens: int | None = None, output_tokens: int | None = None) -> None:
        del input_tokens, output_tokens


class NoTrace:
    @contextmanager
    def model_call(
        self, *, operation: str, provider: str, model: str, input_messages: Any, server_address: str | None = None
    ) -> Iterator[_NoSpan]:
        del operation, provider, model, input_messages, server_address
        yield _NoSpan()


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def summarize(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Run-level metrics; request errors and invalid answers stay in every denominator."""
    predicted = [o.get("prediction", {}).get("intent") for o in outcomes]
    expected = [o["expected"] for o in outcomes]
    pairs = list(zip(expected, predicted, strict=True))
    in_scope = [(e, p) for e, p in pairs if e != OUT_OF_SCOPE]
    predictions = [Prediction.model_validate(o["prediction"]) for o in outcomes if "prediction" in o]
    costs = [p.cost_usd for p in predictions]
    latencies = [o["latency_ms"] for o in outcomes if "latency_ms" in o]
    return {
        "cases": len(outcomes),
        "errors": sum("error" in o for o in outcomes),
        "invalid_answers": sum(p.error is not None for p in predictions),
        "accuracy": _ratio(sum(e == p for e, p in pairs), len(pairs)),
        "in_scope_accuracy": _ratio(sum(e == p for e, p in in_scope), len(in_scope)),
        "oos_recall": _ratio(
            sum(p == OUT_OF_SCOPE for e, p in pairs if e == OUT_OF_SCOPE), len(pairs) - len(in_scope)
        ),
        "oos_precision": _ratio(
            sum(e == OUT_OF_SCOPE for e, p in pairs if p == OUT_OF_SCOPE), sum(p == OUT_OF_SCOPE for p in predicted)
        ),
        "latency_p50_ms": _percentile(latencies, 0.5),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "input_tokens": sum(p.input_tokens or 0 for p in predictions),
        "output_tokens": sum(p.output_tokens or 0 for p in predictions),
        # Only when the provider reports a cost for every call.
        "cost_per_1000_messages_usd": (
            1000 * sum(c for c in costs if c is not None) / len(costs) if costs and None not in costs else None
        ),
    }


async def _evaluate_one(row: Row, *, chat: ChatModel, prompts: Path) -> dict[str, Any]:
    outcome: dict[str, Any] = {"id": row.id, "expected": row.expected.intent}
    started = time.perf_counter()
    try:
        prediction = await route(row.input.text, chat=chat, prompts_dir=prompts, trace=NoTrace())
        outcome["latency_ms"] = round(1000 * (time.perf_counter() - started), 1)
        outcome["prediction"] = prediction.model_dump(mode="json")
    except Exception as exc:
        # Keep errors in the denominator. Do not print provider response
        # bodies or credentials into benchmark artifacts.
        outcome["error"] = type(exc).__name__
    return outcome


async def evaluate(args: argparse.Namespace, chat: ChatModel | None = None) -> dict[str, Any]:
    rows = load_rows(args.data / SPLIT_FILES[args.split])
    if args.limit is not None:
        rows = rows[: args.limit]
    chat = chat or ChatModel(
        client=AsyncOpenAI(timeout=60), provider="openai", model=MODEL, reasoning_effort=REASONING_EFFORT
    )
    slots = asyncio.Semaphore(args.concurrency)

    async def bounded(row: Row) -> dict[str, Any]:
        async with slots:
            return await _evaluate_one(row, chat=chat, prompts=args.prompts)

    outcomes = await asyncio.gather(*(bounded(row) for row in rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(outcome) + "\n" for outcome in outcomes))
    metrics = summarize(outcomes)
    summary = {"model": chat.model, "reasoning_effort": chat.reasoning_effort, "split": args.split, **metrics}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
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
    prep.add_argument("--cache", type=Path, default=Path.home() / ".cache/clinc150-router" / "source-v1")
    run = commands.add_parser("evaluate", help="Call the chat model; uses OPENAI_API_KEY and incurs API charges")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    run.add_argument("--prompts", type=Path, default=Path("prompts"))
    run.add_argument("--limit", type=positive_int)
    run.add_argument("--concurrency", type=positive_int, default=8)
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
