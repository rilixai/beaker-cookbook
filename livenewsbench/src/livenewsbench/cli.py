"""Prepare the dataset and evaluate an agent without Beaker."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import openai
from beaker.tracing import current_trace
from openai import AsyncOpenAI

from livenewsbench import openai_search, youcom
from livenewsbench.chat import ChatModel
from livenewsbench.data import SPLITS, Row, load_rows, prepare
from livenewsbench.grading import GRADER_MODEL, grade
from livenewsbench.runner import (
    MAX_MODEL_CALLS,
    MODEL_ENV,
    SEED_AGENT_DIR,
    SEED_OPENAI_AGENT_DIR,
    Agent,
    OpenAIAgent,
    Rollout,
    list_price_usd,
    load_agent,
    load_openai_agent,
    run_openai_rollout,
    run_rollout,
)
from livenewsbench.tools import WebTools


async def youcom_rollout(question: str, *, args: argparse.Namespace, agent: Agent, key: str) -> Rollout:
    """The agent's model on --base-url (the router by default), with You.com search."""
    on_router = args.base_url == youcom.ROUTER_BASE_URL
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
    async with client, youcom.YouComWeb(key) as backend:
        web = WebTools(backend=backend, trace=current_trace())
        return await run_rollout(question, agent=agent, chat=chat, web=web)


async def openai_rollout(question: str, *, model_name: str, agent: OpenAIAgent) -> Rollout:
    """GPT with OpenAI's built-in web search, using OPENAI_API_KEY."""
    model = openai_search.ResponsesModel(
        client=openai_search.openai_client(), model=model_name, max_calls=MAX_MODEL_CALLS, trace=current_trace()
    )
    async with model.client:
        return await run_openai_rollout(question, agent=agent, model=model)


async def evaluate_row(
    row: Row,
    *,
    rollout_for: Callable[[str], Awaitable[Rollout]],
    priced_model: str | None,
    grader: AsyncOpenAI,
) -> dict[str, Any]:
    outcome: dict[str, Any] = {
        "id": row.id,
        "question": row.input.question,
        "gold": row.expected.answer,
        "human_verified": row.metadata.human_verified,
    }
    # Errors stay in the denominator. Only the exception type is kept: no provider bodies or keys in artifacts.
    try:
        rollout = await rollout_for(row.input.question)
    except Exception as exc:
        outcome.update(error=type(exc).__name__, error_stage="rollout", correct=False)
        return outcome
    outcome.update(
        answer=rollout.answer,
        usage=rollout.usage(),
        list_price_usd=list_price_usd(priced_model, rollout.chat, rollout.web),
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
        "search": args.search,
        "model": args.model,
        "base_url": args.base_url if args.search == "youcom" else f"https://{openai_search.OPENAI_HOST}/v1",
        "split": args.split,
        "cases": len(outcomes),
        "errors": sum("error" in outcome for outcome in outcomes),
        "grading_errors": sum(outcome.get("error_stage") == "grading" for outcome in outcomes),
        "accuracy": correct / len(outcomes),
        "not_attempted": sum(outcome.get("grade") == "NOT_ATTEMPTED" for outcome in outcomes) / len(outcomes),
        # Answers withheld because the question opened, searched in or cited a blocked host (OpenAI search only).
        "withheld_blocked_host": sum(outcome["usage"]["finish"] == "blocked_host" for outcome in done),
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


def rollout_runner(args: argparse.Namespace) -> tuple[Callable[[str], Awaitable[Rollout]], str | None]:
    """How to run one question under --search, and the model to price (None: no recorded price)."""
    if args.search == "openai":
        openai_agent = load_openai_agent(args.agent or SEED_OPENAI_AGENT_DIR)

        def run_openai(question: str) -> Awaitable[Rollout]:
            return openai_rollout(question, model_name=args.model, agent=openai_agent)

        return run_openai, args.model
    agent = load_agent(args.agent or SEED_AGENT_DIR)
    key = youcom.api_key()

    def run_youcom(question: str) -> Awaitable[Rollout]:
        return youcom_rollout(question, args=args, agent=agent, key=key)

    return run_youcom, args.model if args.base_url == youcom.ROUTER_BASE_URL else None


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(args.data, args.split)[: args.limit]
    rollout_for, priced_model = rollout_runner(args)
    semaphore = asyncio.Semaphore(args.concurrency)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outcomes: list[dict[str, Any]] = []
    async with AsyncOpenAI(max_retries=5) as grader:
        await check_grader(grader)

        async def one(row: Row) -> dict[str, Any]:
            async with semaphore:
                return await evaluate_row(row, rollout_for=rollout_for, priced_model=priced_model, grader=grader)

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
    prep.add_argument(
        "--train-sample",
        type=positive_int,
        help="Keep this many train questions, stratified by month and category (default: all 612)",
    )
    prep.add_argument("--seed", type=int, default=0, help="Seed that fixes which questions --train-sample keeps")
    run = commands.add_parser(
        "evaluate", help="Run and grade the agent; uses YDC_API_KEY and OPENAI_API_KEY and incurs API charges"
    )
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--split", choices=SPLITS, default="val")
    run.add_argument("--limit", type=positive_int)
    run.add_argument(
        "--search",
        choices=("youcom", "openai"),
        default="youcom",
        help="youcom: the model on --base-url with You.com search; openai: GPT with OpenAI's built-in web search",
    )
    run.add_argument("--agent", type=Path, help="Agent package directory, a candidate's (default: the seed)")
    run.add_argument("--model", help=f"Default: ${MODEL_ENV}, else deepseek-v4.1-flash, or gpt-6-luna with openai")
    run.add_argument("--base-url", default=youcom.ROUTER_BASE_URL, help="youcom only: endpoint for the agent's model")
    run.add_argument("--api-key-env", default=youcom.API_KEY_ENV, help="youcom only: variable holding its key")
    run.add_argument("--concurrency", type=positive_int, default=8)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.output, args.cache, train_sample=args.train_sample, seed=args.seed), indent=2))
        return
    default_model = openai_search.DEFAULT_MODEL if args.search == "openai" else youcom.DEFAULT_MODEL
    args.model = args.model or os.environ.get(MODEL_ENV) or default_model
    summary = asyncio.run(evaluate(args))
    print(json.dumps(summary, indent=2))
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
