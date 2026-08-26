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
        }

    return {
        "domains": {d: _agg(rs) for d, rs in sorted(by_domain.items())},
        "overall": _agg(results),
    }


def format_summary(summary: dict[str, Any]) -> str:
    lines = [f"{'domain':<12} {'tasks':>5} {'pass_rate':>10} {'partial_credit':>15}"]
    for domain, agg in summary["domains"].items():
        lines.append(f"{domain:<12} {agg['tasks']:>5} {agg['pass_rate']:>10.3f} {agg['partial_credit']:>15.3f}")
    overall = summary["overall"]
    lines.append(
        f"{'overall':<12} {overall['tasks']:>5} {overall['pass_rate']:>10.3f} {overall['partial_credit']:>15.3f}"
    )
    return "\n".join(lines)
