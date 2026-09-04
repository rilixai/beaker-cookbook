"""CLI: `automationbench-skills run` executes rollouts and writes per-task
JSON files; `automationbench-skills evaluate` aggregates them."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from automationbench_skills.evaluation.summary import format_summary, summarize
from automationbench_skills.runner import DEFAULT_MAX_STEPS, DEFAULT_MODEL, ModelSpec, RunResult, run_split


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--skills-dir", type=Path, default=None, help="Skills directory of SKILL.md folders (read live)")
    p.add_argument("--no-skills", action="store_true", help="Baseline arm: no skill tools registered")
    p.add_argument(
        "--prompts-dir",
        type=Path,
        default=None,
        help="Directory whose system.md replaces the benchmark system prompt (read live)",
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--base-url", default=None, help="OpenAI-compatible gateway base URL")
    p.add_argument("--api-key-var", default="OPENAI_API_KEY")
    p.add_argument("--api", default="auto", help="auto|chat_completions|responses|anthropic|gemini_interactions")
    p.add_argument("--toolset", choices=["zapier", "api"], default="zapier")
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--max-concurrent", type=int, default=8)
    p.add_argument(
        "--task-timeout", type=float, default=None, help="Per-task rollout timeout in seconds (scores 0 on expiry)"
    )
    p.add_argument("--limit", type=int, default=None, help="Run only the first N tasks of the split")
    p.add_argument("--output-dir", type=Path, default=None, help="Default: runs/<split>-<timestamp>")


def _cmd_run(args: argparse.Namespace) -> int:
    from automationbench_skills.data import load_split

    if args.skills_dir is not None and args.no_skills:
        print("error: --skills-dir and --no-skills are mutually exclusive", file=sys.stderr)
        return 2
    skills_dir: Path | None = args.skills_dir
    if skills_dir is None and not args.no_skills:
        print("note: no --skills-dir given; running baseline (same as --no-skills)")
    if skills_dir is not None and not skills_dir.is_dir():
        print(f"error: --skills-dir {skills_dir} is not a directory", file=sys.stderr)
        return 2
    prompts_dir: Path | None = args.prompts_dir
    if prompts_dir is not None and skills_dir is None:
        print("error: --prompts-dir replaces the benchmark prompt; the no-skills baseline keeps it", file=sys.stderr)
        return 2
    if prompts_dir is not None and not prompts_dir.is_dir():
        print(f"error: --prompts-dir {prompts_dir} is not a directory", file=sys.stderr)
        return 2

    samples = load_split(args.split)
    if args.limit is not None:
        samples = samples[: args.limit]

    output_dir = args.output_dir or Path("runs") / f"{args.split}-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = ModelSpec(
        name=args.model,
        base_url=args.base_url,
        api_key_var=args.api_key_var,
        api=args.api,
        reasoning_effort=args.reasoning_effort,
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "limit": args.limit,
                "model": model.name,
                "base_url": model.base_url,
                "api": model.api,
                "reasoning_effort": model.reasoning_effort,
                "toolset": args.toolset,
                "max_steps": args.max_steps,
                "skills_dir": str(skills_dir) if skills_dir else None,
                "prompts_dir": str(prompts_dir) if prompts_dir else None,
                "tasks": [s.task_name for s in samples],
            },
            indent=2,
        )
    )
    print(f"Running {len(samples)} task(s) from split '{args.split}' with model {model.name} -> {output_dir}")

    def on_result(result: RunResult) -> None:
        path = output_dir / f"{result.task_name}.json"
        path.write_text(json.dumps(result.to_json(), indent=2, default=str))
        print(
            f"  {result.task_name}: partial_credit={result.partial_credit:.3f} "
            f"passed={bool(result.task_completed_correctly)}"
        )

    results = run_split(
        samples,
        model=model,
        skills_dir=skills_dir,
        prompts_dir=prompts_dir,
        toolset=args.toolset,
        max_steps=args.max_steps,
        max_concurrent=args.max_concurrent,
        timeout=args.task_timeout,
        on_result=on_result,
    )
    summary = summarize([r.to_json() for r in results])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print(format_summary(summary))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    results: list[dict[str, Any]] = []
    for path in sorted(args.output_dir.glob("*.json")):
        if path.name in ("config.json", "summary.json"):
            continue
        data = json.loads(path.read_text())
        if "task_completed_correctly" in data:
            results.append(data)
    if not results:
        print(f"error: no per-task result files found in {args.output_dir}", file=sys.stderr)
        return 1
    summary = summarize(results)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(format_summary(summary))
    return 0


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="automationbench-skills")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="Run rollouts on a frozen split and write per-task results")
    _add_run_args(run_p)
    run_p.set_defaults(func=_cmd_run)
    eval_p = sub.add_parser("evaluate", help="Aggregate per-task results into per-domain/overall metrics")
    eval_p.add_argument("--output-dir", type=Path, required=True)
    eval_p.set_defaults(func=_cmd_evaluate)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
