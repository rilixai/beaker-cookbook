"""Prepare splits and evaluate the agent without Beaker."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from tau2_bench import benchmark, scoring
from tau2_bench.agent import Agent, load_instructions
from tau2_bench.data import DOMAINS, SPLIT_FILES, load_rows, prepare


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    rows = load_rows(args.data / SPLIT_FILES[args.split])
    if args.domain:
        rows = [row for row in rows if row.input.domain == args.domain]
    if args.limit is not None:
        rows = rows[: args.limit]
    instructions = load_instructions(args.prompts)
    chat = benchmark.default_chat_model()
    rewards: dict[str, list[float]] = {domain: [] for domain in DOMAINS}
    turns: list[int] = []
    errors = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for row in rows:
            domain, task_id = row.input.domain, row.input.task_id
            outcome: dict[str, object] = {"id": row.id}
            value = 0.0
            try:
                simulation = benchmark.run_task(
                    domain,
                    task_id,
                    lambda tools, policy: Agent(tools, policy, chat=chat, instructions=instructions),
                    agent_model=chat.model,
                )
                reward = benchmark.evaluate(simulation, domain, task_id)
                scored = scoring.score(reward, simulation, benchmark.reward_basis(domain, task_id))
                value = scored.objective
                outcome |= {"fields": dict(scored.field_scores), "summary": scoring.summary(simulation)}
                turns.append(sum(message.role == "assistant" for message in simulation.messages))
            except Exception as exc:
                # Errors stay in the denominator as failed cases.
                outcome["error"] = type(exc).__name__
                errors += 1
            outcome["reward"] = value
            rewards[domain].append(value)
            print(f"{row.id}: reward={value}" + (f" error={outcome['error']}" if "error" in outcome else ""))
            output.write(json.dumps(outcome) + "\n")
            output.flush()
    all_rewards = [r for values in rewards.values() for r in values]
    metrics: dict[str, object] = {
        "model": chat.model,
        "split": args.split,
        "cases": len(all_rewards),
        "errors": errors,
        "reward": statistics.fmean(all_rewards) if all_rewards else 0.0,
        "reward_by_domain": {d: statistics.fmean(v) for d, v in rewards.items() if v},
        "mean_agent_turns": statistics.fmean(turns) if turns else None,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Download pinned upstream data and write splits")
    prep.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("evaluate", help="Run full conversations; uses OPENAI_API_KEY and incurs API charges")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--split", choices=tuple(SPLIT_FILES), default="validation")
    run.add_argument("--domain", choices=DOMAINS)
    run.add_argument("--prompts", type=Path, default=Path("prompts"))
    run.add_argument("--limit", type=positive_int)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.output), indent=2))
    else:
        metrics = evaluate(args)
        print(json.dumps(metrics, indent=2))
        if metrics["errors"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
