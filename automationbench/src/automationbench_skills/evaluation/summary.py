"""Summary metrics: pass rate (mean task_completed_correctly) AND mean
partial_credit, per domain and overall. Both metrics, always."""

from __future__ import annotations

from typing import Any


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-task result dicts (RunResult.to_json shape) into per-domain
    and overall pass rate + mean partial credit."""
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_domain.setdefault(r["domain"], []).append(r)

    def _agg(rs: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rs)
        return {
            "tasks": n,
            "pass_rate": sum(float(r["task_completed_correctly"]) for r in rs) / n if n else 0.0,
            "partial_credit": sum(float(r["partial_credit"]) for r in rs) / n if n else 0.0,
            "avg_latency_s": (
                sum(float(r["latency_s"]) for r in rs if r.get("latency_s") is not None)
                / sum(1 for r in rs if r.get("latency_s") is not None)
                if any(r.get("latency_s") is not None for r in rs)
                else None
            ),
            "avg_cost_usd": (
                sum(float(r["cost_usd"]) for r in rs if r.get("cost_usd") is not None)
                / sum(1 for r in rs if r.get("cost_usd") is not None)
                if any(r.get("cost_usd") is not None for r in rs)
                else None
            ),
        }

    return {
        "domains": {d: _agg(rs) for d, rs in sorted(by_domain.items())},
        "overall": _agg(results),
    }


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"{'domain':<12} {'tasks':>5} {'pass_rate':>10} {'partial_credit':>15} "
        f"{'avg_latency_s':>14} {'avg_cost_usd':>13}"
    ]
    for domain, agg in summary["domains"].items():
        latency = "-" if agg["avg_latency_s"] is None else f"{agg['avg_latency_s']:.1f}"
        cost = "-" if agg["avg_cost_usd"] is None else f"{agg['avg_cost_usd']:.4f}"
        lines.append(
            f"{domain:<12} {agg['tasks']:>5} {agg['pass_rate']:>10.3f} {agg['partial_credit']:>15.3f} "
            f"{latency:>14} {cost:>13}"
        )
    overall = summary["overall"]
    latency = "-" if overall["avg_latency_s"] is None else f"{overall['avg_latency_s']:.1f}"
    cost = "-" if overall["avg_cost_usd"] is None else f"{overall['avg_cost_usd']:.4f}"
    lines.append(
        f"{'overall':<12} {overall['tasks']:>5} {overall['pass_rate']:>10.3f} {overall['partial_credit']:>15.3f} "
        f"{latency:>14} {cost:>13}"
    )
    return "\n".join(lines)
