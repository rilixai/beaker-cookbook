"""Prepare the dataset and evaluate an agent without Beaker."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import openai
from beaker.tracing import current_trace
from openai import AsyncOpenAI

from livenewsbench import youcom
from livenewsbench.chat import ChatModel
from livenewsbench.data import SPLITS, Row, load_rows, prepare
from livenewsbench.grading import GRADER_MODEL, grade
from livenewsbench.runner import (
    MAX_MODEL_CALLS,
    MODEL_ENV,
    SEED_AGENT_DIR,
    Agent,
    list_price_usd,
    load_agent,
    run_rollout,
)
from livenewsbench.tools import WebTools


async def evaluate_row(
    row: Row, *, args: argparse.Namespace, agent: Agent, key: str, grader: AsyncOpenAI
) -> dict[str, Any]:
    on_router = args.base_url == youcom.ROUTER_BASE_URL
    outcome: dict[str, Any] = {
        "id": row.id,
        "question": row.input.question,
        "gold": row.expected.answer,
        "human_verified": row.metadata.human_verified,
    }
    client = AsyncOpenAI(
        base_url=args.base_url, api_key=os.environ.get(args.api_key_env, ""), max_retries=5, timeout=300
    )
    chat = ChatModel(
        client=client,
        model=args.model,
        provider="youcom" if on_router else "openai",
        server_address=urlsplit(args.base_url).hostname or "",
        max_calls=MAX_MODEL_CALLS,
        trace=current_trace(),
        request_options=youcom.REQUEST_OPTIONS if on_router else {},
    )
    # Errors stay in the denominator. Only the exception type is kept: no provider bodies or keys in artifacts.
    try:
        async with client, youcom.YouComWeb(key) as backend:
            web = WebTools(backend=backend, trace=current_trace())
            rollout = await run_rollout(row.input.question, agent=agent, chat=chat, web=web)
    except Exception as exc:
        outcome.update(error=type(exc).__name__, error_stage="rollout", correct=False)
        return outcome
    outcome.update(
        answer=rollout.answer,
        usage=rollout.usage(),
        list_price_usd=list_price_usd(args.model if on_router else None, rollout.chat, rollout.web),
    )
    try:
        verdict = await grade(
            grader, model=GRADER_MODEL, question=row.input.question, gold=row.expected.answer, predicted=rollout.answer
        )
    except Exception as exc:
        outcome.update(error=type(exc).__name__, error_stage="grading", correct=False)
        return outcome
    outcome.update(grade=verdict, correct=verdict == "CORRECT")
    return outcome


def summarize(outcomes: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    # Latency and cost cover every finished rollout, including ones the grader then failed on.
    done = [outcome for outcome in outcomes if "usage" in outcome]
    correct = sum(outcome["correct"] for outcome in outcomes)
    seconds = [outcome["usage"]["seconds"] for outcome in done]
    costs = [outcome["list_price_usd"]["total"] for outcome in done]

    def mean(values: list[float]) -> float | None:
        return statistics.fmean(values) if values else None

    total_cost = sum(costs) if costs and None not in costs else None
    return {
        "model": args.model,
        "base_url": args.base_url,
        "split": args.split,
        "cases": len(outcomes),
        "errors": sum("error" in outcome for outcome in outcomes),
        "grading_errors": sum(outcome.get("error_stage") == "grading" for outcome in outcomes),
        "accuracy": correct / len(outcomes),
        "not_attempted": sum(outcome.get("grade") == "NOT_ATTEMPTED" for outcome in outcomes) / len(outcomes),
        "seconds_mean": mean(seconds),
        "seconds_p50": statistics.median(seconds) if seconds else None,
        "seconds_p90": statistics.quantiles(seconds, n=10)[-1] if len(seconds) >= 2 else None,
        "model_calls_mean": mean([outcome["usage"]["model"]["calls"] for outcome in done]),
        "searches_mean": mean([outcome["usage"]["web"]["searches"] for outcome in done]),
        "page_visits_mean": mean([outcome["usage"]["web"]["page_visits"] for outcome in done]),
        "list_price_usd_total": total_cost,
        "list_price_usd_per_question": None if total_cost is None else total_cost / len(done),
        "list_price_usd_per_correct": None if total_cost is None or not correct else total_cost / correct,
    }


async def check_grader(grader: AsyncOpenAI) -> None:
    """Stop before any paid rollout if the grader's key is rejected."""
    try:
        await grader.models.retrieve(GRADER_MODEL)
    except openai.AuthenticationError:
        raise SystemExit(f"The grader ({GRADER_MODEL}) rejected OPENAI_API_KEY (HTTP 401). No rollouts ran.") from None


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(args.data, args.split)[: args.limit]
    agent = load_agent(args.agent)
    key = youcom.api_key()
    semaphore = asyncio.Semaphore(args.concurrency)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outcomes: list[dict[str, Any]] = []
    async with AsyncOpenAI(max_retries=5) as grader:
        await check_grader(grader)

        async def one(row: Row) -> dict[str, Any]:
            async with semaphore:
                return await evaluate_row(row, args=args, agent=agent, key=key, grader=grader)

        with args.output.open("w") as output:
            for finished in asyncio.as_completed([one(row) for row in rows]):
                outcome = await finished
                outcomes.append(outcome)
                output.write(json.dumps(outcome) + "\n")
                output.flush()
    summary = summarize(outcomes, args)
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
    prep = commands.add_parser("prepare", help="Download and verify the pinned LiveNewsBench release")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--cache", type=Path, default=Path.home() / ".cache/livenewsbench/jan_2026_release_2")
    run = commands.add_parser(
        "evaluate", help="Run and grade the agent; uses YDC_API_KEY and OPENAI_API_KEY and incurs API charges"
    )
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--split", choices=SPLITS, default="val")
    run.add_argument("--limit", type=positive_int)
    run.add_argument("--agent", type=Path, default=SEED_AGENT_DIR, help="Agent package directory (a candidate's)")
    run.add_argument("--model", default=os.environ.get(MODEL_ENV) or youcom.DEFAULT_MODEL)
    run.add_argument("--base-url", default=youcom.ROUTER_BASE_URL, help="OpenAI-compatible endpoint for the agent")
    run.add_argument("--api-key-env", default=youcom.API_KEY_ENV, help="Environment variable holding its key")
    run.add_argument("--concurrency", type=positive_int, default=8)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.output, args.cache), indent=2))
        return
    summary = asyncio.run(evaluate(args))
    print(json.dumps(summary, indent=2))
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
