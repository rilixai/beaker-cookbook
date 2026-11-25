"""Prepare the pinned data, evaluate the agent, or play one game, without Beaker."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from alfworld_agent import agent
from alfworld_agent.data import CACHE, SPLIT_FILES, Row, find_row, games_root, load_rows, prepare
from alfworld_agent.decide import (
    DEFAULT_TARGET,
    PRICE_PER_MILLION_INPUT_TOKENS,
    ChatModel,
    DecisionTarget,
    NullTracer,
    Session,
    default_chat_model,
)
from alfworld_agent.env import Episode, run_episode
from alfworld_agent.scoring import score, summarize


async def play(
    row: Row,
    root: Path,
    instructions: str,
    *,
    target: DecisionTarget = DEFAULT_TARGET,
    chat_factory: Callable[[], ChatModel] = default_chat_model,
    client: Any = None,
) -> Episode:
    policy = agent.Agent(instructions)
    async with Session(target=target, chat_factory=chat_factory, client=client) as session:
        return await run_episode(row.id, root / row.input.game, act=policy.act, session=session, trace=NullTracer())


def case_record(row: Row, episode: Episode) -> dict[str, Any]:
    output = episode.output()
    scored = score(output)
    return {
        "id": row.id,
        "task_type": row.expected.task_type,
        "success": scored.success,
        "steps": episode.steps,
        "hit_step_limit": episode.hit_step_limit,
        "error": episode.error,
        "fields": scored.field_scores,
        "decision_calls": output["decision_calls"],
        "chat_calls": output["chat_calls"],
        "repeats": episode.repeats,
        "decision_input_tokens": output["decision_input_tokens"],
        "decision_latencies_s": episode.context()["decision_latencies_s"],
        "latency_s": episode.latency_s,
        "models": output["models"],
        "goal": episode.goal,
        "transcript": episode.transcript,
    }


async def evaluate(
    args: argparse.Namespace, *, chat_factory: Callable[[], ChatModel] = default_chat_model, client: Any = None
) -> dict[str, Any]:
    rows = load_rows(args.data / SPLIT_FILES[args.split])
    if args.limit is not None:
        rows = rows[: args.limit]
    root = games_root(args.cache)
    instructions = agent.load_instructions(args.prompts)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(row: Row) -> Episode:
        async with semaphore:
            return await play(row, root, instructions, chat_factory=chat_factory, client=client)

    episodes = await asyncio.gather(*(one(row) for row in rows))
    records = [case_record(row, episode) for row, episode in zip(rows, episodes, strict=True)]
    summary: dict[str, Any] = {
        "model": DEFAULT_TARGET.model,
        "split": args.split,
        "prompts": str(args.prompts),
        **summarize(records, price_per_million_input_tokens=PRICE_PER_MILLION_INPUT_TOKENS),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def print_transcript(episode: Episode) -> None:
    print(f"Task: {episode.goal}\n")
    for step in episode.transcript:
        decisions = step.get("decisions") or []
        top = max(decisions[-1]["probabilities"].values()) if decisions else None
        print(f"[{step['step']}] {len(step['options'])} options" + (f", p={top:.2f}" if top is not None else ""))
        print(f"  > {step.get('action')}")
        if "result" in step:
            print(f"  {step['result']}")
    outcome = episode.output()["outcome"]
    print(f"\nwon={outcome['won']} steps={outcome['steps']} error={outcome['error']}")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE, help="where pinned downloads and game files are kept")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Download and verify the pinned game files; write splits and manifest")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument(
        "--games-per-type", type=positive_int, help="keep the first N games of each task type per split (small runs)"
    )
    run = commands.add_parser("evaluate", help="Run the agent; uses TYPESAFE_API_KEY and incurs API charges")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--split", choices=tuple(SPLIT_FILES), default="validation")
    run.add_argument("--prompts", type=Path, default=Path("prompts"))
    run.add_argument("--limit", type=positive_int)
    run.add_argument("--concurrency", type=positive_int, default=8)
    run.add_argument("--output", type=Path, required=True)
    one = commands.add_parser("play", help="Play one game and print the transcript; incurs API charges")
    one.add_argument("--data", type=Path, required=True)
    one.add_argument("--game", required=True, help="case ID, e.g. alfworld-valid_seen-0123456789ab")
    one.add_argument("--prompts", type=Path, default=Path("prompts"))
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.output, args.cache, games_per_type=args.games_per_type), indent=2))
        return
    if args.command == "play":
        row = find_row(args.data, args.game)
        episode = asyncio.run(play(row, games_root(args.cache), agent.load_instructions(args.prompts)))
        print_transcript(episode)
        if episode.error:
            raise SystemExit(1)
        return
    summary = asyncio.run(evaluate(args))
    print(json.dumps(summary, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
