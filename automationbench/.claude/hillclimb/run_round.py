"""Hillclimb runner: the 54-case quickstart set (6 train + 3 test per domain,
same selection as .beaker/upload_splits.py) x R reps, written in the
hillclimb layout. Resume-safe at (case, rep).

usage: uv run python .claude/hillclimb/run_round.py <variant> [--reps R] [--limit N] [--ids a,b]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from automationbench_skills.data.tasks import PUBLIC_DOMAINS, load_split
from automationbench_skills.runner import ModelSpec, run_one_async


ROOT = Path(__file__).resolve().parents[2]  # automationbench/
FLOW = Path(__file__).resolve().parent / "automationbench"
TRAIN_PER_DOMAIN, TEST_PER_DOMAIN = 6, 3


def quickstart() -> tuple[list, list]:
    def take(split: str, k: int) -> list:
        by: dict[str, list] = defaultdict(list)
        for s in load_split(split):
            by[s.domain].append(s)
        return [s for d in PUBLIC_DOMAINS for s in by[d][:k]]

    return take("train", TRAIN_PER_DOMAIN), take("test", TEST_PER_DOMAIN)


def user_prompt(sample) -> str:
    return "\n\n".join(str(m.get("content") or "") for m in sample.prompt if m.get("role") == "user").strip()


def to_turns(traj: list[dict]) -> list[dict]:
    turns = []
    for m in traj:
        role = m.get("role")
        if role == "assistant":
            content = m.get("content") or ""
            if isinstance(content, list):
                content = "\n".join(str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content)
            if content:
                turns.append({"role": "assistant", "content": content})
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, str):
                    try:
                        tc = json.loads(tc)
                    except ValueError:
                        tc = {"name": "?", "arguments": tc}
                fn = tc.get("function") or tc
                turns.append({"role": "tool_call", "name": fn.get("name"), "content": str(fn.get("arguments"))})
        elif role == "tool":
            turns.append({"role": "tool_result", "name": m.get("name"), "content": str(m.get("content"))})
        else:
            turns.append({"role": role or "user", "content": str(m.get("content"))})
    return turns


async def main() -> int:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--api", default="responses")
    ap.add_argument("--max-concurrent", type=int, default=16)
    ap.add_argument("--task-timeout", type=float, default=900)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--skills-dir", default=str(ROOT / "skills"))
    ap.add_argument("--prompts-dir", default=str(ROOT / "prompts"))
    a = ap.parse_args()

    train, test = quickstart()
    split_of = {s.task_name: "train" for s in train} | {s.task_name: "test" for s in test}
    samples = train + test
    if a.ids:
        want = set(a.ids.split(","))
        samples = [s for s in samples if s.task_name in want]
    if a.limit:
        samples = samples[: a.limit]

    out = FLOW / a.variant
    (out / "traces").mkdir(parents=True, exist_ok=True)
    res_path, err_path = out / "results.jsonl", out / "errors.jsonl"
    done = set()
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["prompt_id"], r["rep"]))
    jobs = [(s, k) for s in samples for k in range(a.reps) if (s.task_name, k) not in done]
    model = ModelSpec(name=a.model, reasoning_effort=a.reasoning_effort, api=a.api)
    print(
        f"[{a.variant}] {len(samples)} cases x {a.reps} reps, {len(jobs)} to run, model={a.model} "
        f"effort={a.reasoning_effort} skills={a.skills_dir}",
        flush=True,
    )

    sem = asyncio.Semaphore(a.max_concurrent)
    n_done, t0 = 0, time.monotonic()

    async def one(sample, rep):
        nonlocal n_done
        async with sem:
            try:
                r = await run_one_async(
                    sample, model=model, skills_dir=a.skills_dir, prompts_dir=a.prompts_dir, timeout=a.task_timeout
                )
            except Exception as e:  # harness error: record, don't score
                with err_path.open("a") as f:
                    f.write(json.dumps({"prompt_id": sample.task_name, "rep": rep, "error": repr(e)}) + "\n")
                print(f"  ERR {sample.task_name} rep{rep}: {e!r}", flush=True)
                return
        if r.error and not r.trajectory:  # model/infra failure before any action: harness error, unscored
            with err_path.open("a") as f:
                f.write(
                    json.dumps({"prompt_id": sample.task_name, "rep": rep, "error": str(r.error), "usage": r.usage})
                    + "\n"
                )
            print(f"  ERR {sample.task_name} rep{rep}: {str(r.error)[:300]}", flush=True)
            return
        trace_rel = f"{a.variant}/traces/{sample.task_name}_rep{rep}.json"
        system = next(
            (m.get("content") for m in r.raw.get("prompt", []) if isinstance(m, dict) and m.get("role") == "system"),
            None,
        )
        turns = (
            ([{"role": "system", "content": str(system)}] if system else [])
            + [{"role": "user", "content": user_prompt(sample)}]
            + to_turns(r.trajectory)
        )
        (FLOW / trace_rel).write_text(json.dumps(turns, indent=1, default=str))
        (out / "traces" / f"{sample.task_name}_rep{rep}.assertions.json").write_text(
            json.dumps(r.assertion_results, indent=1, default=str)
        )
        n_tool = sum(1 for t in turns if t["role"] == "tool_call")
        n_skill = sum(1 for t in turns if t["role"] == "tool_call" and t.get("name") in ("read_skill", "list_skills"))
        row = {
            "prompt_id": sample.task_name,
            "rep": rep,
            "prompt": user_prompt(sample),
            "tags": [sample.domain, split_of[sample.task_name]],
            "grade": {
                "score": 0.8 * r.partial_credit + 0.2 * r.task_completed_correctly,
                "partial_credit": r.partial_credit,
                "pass": r.task_completed_correctly,
            },
            "model": r.raw.get("model") or a.model,
            "latency_s": r.latency_s,
            "cost_usd": r.cost_usd,
            "usage": r.usage,
            "tool_calls": n_tool,
            "skill_calls": n_skill,
            "error": str(r.error) if r.error else None,
        }
        with res_path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        n_done += 1
        el = time.monotonic() - t0
        msg = (
            f"  {n_done}/{len(jobs)} done  {sample.task_name} rep{rep} pc={r.partial_credit:.2f} "
            f"pass={int(r.task_completed_correctly)}  ~{el / n_done * (len(jobs) - n_done):.0f}s left"
        )
        print(msg, flush=True)
        (out / "progress.txt").write_text(msg + "\n")

    await asyncio.gather(*(one(s, k) for s, k in jobs))
    if not (out / "summary.json").exists():
        (out / "summary.json").write_text(json.dumps({"description": a.variant, "target": "skill"}))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
