"""Aggregation over per-question results and resume-from-manifest selection.

Headline metric: correctness at 0.0% allowable absolute relative error
(report default); 0.1%, 1.0% and 5.0% are reported alongside. Timeouts and
errored questions score 0 and are **included** in every aggregate.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import TYPE_CHECKING, Any

from officeqa import config
from officeqa.data.dataset import EvalRecord


if TYPE_CHECKING:
    from officeqa.runner import RunResult


def _tol_label(t: float) -> str:
    return f"{t * 100:g}%"


def summarize(results: Sequence[RunResult], *, expected: Sequence[str] | None = None) -> dict[str, Any]:
    """Aggregate metrics. ``expected`` (uids) lets missing questions count as 0."""
    by_uid = {r.uid: r for r in results}
    uids = list(expected) if expected is not None else sorted(by_uid)
    n = len(uids)
    present = [by_uid[u] for u in uids if u in by_uid]
    missing = [u for u in uids if u not in by_uid]

    correctness = {}
    for t in config.TOLERANCES:
        hits = sum(r.scores.get(str(t), 0.0) for r in present)
        correctness[_tol_label(t)] = hits / n if n else 0.0

    statuses = Counter(r.status for r in present)
    statuses["missing"] = len(missing)
    costs = [r.cost_usd for r in present if r.cost_usd is not None]
    return {
        "n": n,
        "n_scored": len(present),
        "missing": missing,
        "correctness": correctness,
        "headline_tolerance": _tol_label(config.HEADLINE_TOLERANCE),
        "correct": int(sum(r.correct for r in present)),
        "statuses": dict(statuses),
        "mean_latency_s": mean([r.latency_s for r in present]) if present else 0.0,
        "mean_queue_wait_s": mean([r.queue_wait_s for r in present]) if present else 0.0,
        "mean_tool_calls": mean([r.tool_calls for r in present]) if present else 0.0,
        "mean_steps": mean([r.steps for r in present]) if present else 0.0,
        "total_cost_usd": sum(costs) if costs else None,
        "mean_cost_usd": (sum(costs) / len(costs)) if costs else None,
        "cost_known_for": len(costs),
        "total_tokens": sum(
            int(r.usage.get("prompt_tokens", 0)) + int(r.usage.get("completion_tokens", 0)) for r in present
        ),
        "attempts": sum(r.attempts for r in present),
        "tool_call_counts": dict(sum((Counter(r.tool_call_counts) for r in present), Counter())),
    }


def select_to_run(
    records: Sequence[EvalRecord], existing: Mapping[str, RunResult], *, rerun: bool = False
) -> list[EvalRecord]:
    """Resume policy: reuse questions that completed cleanly, re-run the rest; ``rerun`` forces all."""
    if rerun:
        return list(records)
    return [r for r in records if r.uid not in existing or not existing[r.uid].clean]


def format_summary(summary: Mapping[str, Any]) -> str:
    lines = [f"n={summary['n']}  scored={summary['n_scored']}  statuses={summary['statuses']}"]
    corr = summary["correctness"]
    lines.append("correctness: " + "  ".join(f"@{k} {v * 100:.1f}%" for k, v in corr.items()))
    cost = summary["total_cost_usd"]
    lines.append(
        f"mean latency {summary['mean_latency_s'] / 60:.1f} min  mean tool calls {summary['mean_tool_calls']:.1f}  "
        f"mean steps {summary['mean_steps']:.1f}  total cost {'$%.2f' % cost if cost is not None else 'unknown'}"
    )
    return "\n".join(lines)
