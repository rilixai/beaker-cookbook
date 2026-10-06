"""Recompute per-variant train/test numbers from raw results.jsonl (case-mean over reps, then mean over cases)."""
import json, math, sys
from collections import defaultdict
from pathlib import Path

FLOW = Path(__file__).resolve().parent / "automationbench"
st = json.loads((FLOW / "_state.json").read_text())
split = {i: "train" for i in st["train_ids"]} | {i: "test" for i in st["test_ids"]}
variants = sys.argv[1:] or sorted(p.name for p in FLOW.iterdir() if (p / "results.jsonl").exists())
base = None
print(f"{'variant':10} {'n':>4} {'test score':>16} {'train score':>16} {'test pass':>9} {'test pc':>8} {'tr pass':>8} {'tr pc':>6} {'Mtok in':>8} {'tools':>6} {'skills':>6} {'errs':>4}")
for v in variants:
    rows = [json.loads(l) for l in (FLOW / v / "results.jsonl").read_text().splitlines()]
    errs = (FLOW / v / "errors.jsonl")
    nerr = len(errs.read_text().splitlines()) if errs.exists() else 0
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k, x in r["grade"].items():
            per[r["prompt_id"]][k].append(x)
    def agg(sp, k):
        xs = [sum(d[k]) / len(d[k]) for c, d in per.items() if split.get(c) == sp]
        if not xs: return float("nan"), float("nan")
        m = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1))
        return m, 1.96 * sd / math.sqrt(len(xs))
    ts, tci = agg("test", "score"); rs, rci = agg("train", "score")
    tin = sum(r["usage"].get("input_tokens", 0) for r in rows) / 1e6
    tools = sum(r["tool_calls"] for r in rows) / len(rows); sk = sum(r["skill_calls"] for r in rows) / len(rows)
    print(f"{v:10} {len(rows):>4} {ts:7.3f} ±{tci:.3f}     {rs:7.3f} ±{rci:.3f}     {agg('test','pass')[0]:9.3f} {agg('test','partial_credit')[0]:8.3f} {agg('train','pass')[0]:8.3f} {agg('train','partial_credit')[0]:6.3f} {tin:8.1f} {tools:6.1f} {sk:6.1f} {nerr:>4}")
