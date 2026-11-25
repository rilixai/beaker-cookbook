"""Score one episode and summarize a run. Fixed: not part of the editable surface.

The objective is task success: 1 if the game is won within the step limit,
else 0. ALFWorld's text env reports no partial progress (its score is 0 until
the game is won), so there is no partial credit. Field scores must lie in
[0, 1], so step and call counts are scaled; the raw counts are in the output.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from alfworld_agent.env import STEP_LIMIT


# field_scores["decision_calls"] = calls / DECISION_CALLS_SCALE, capped at 1 (two calls per step allowed).
DECISION_CALLS_SCALE = 2 * STEP_LIMIT


class Outcome(BaseModel):
    model_config = ConfigDict(extra="ignore")

    won: bool
    steps: int
    hit_step_limit: bool
    error: str | None


class EpisodeOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    outcome: Outcome
    decision_calls: int
    repeats: int
    mean_top_probability: float | None


@dataclass(frozen=True)
class Scored:
    objective: float
    field_scores: dict[str, float]
    success: bool
    hit_step_limit: bool
    error: str | None


def score(output: Any) -> Scored:
    try:
        episode = EpisodeOutput.model_validate(output)
    except ValidationError:
        return Scored(0.0, {"success": 0.0}, False, False, "malformed_output")
    outcome = episode.outcome
    success = outcome.won and outcome.error is None and outcome.steps <= STEP_LIMIT
    steps = outcome.steps
    return Scored(
        objective=float(success),
        field_scores={
            "success": float(success),
            "steps": min(steps / STEP_LIMIT, 1.0),
            "decision_calls": min(episode.decision_calls / DECISION_CALLS_SCALE, 1.0),
            "repeat_rate": episode.repeats / steps if steps else 0.0,
            "mean_top_probability": episode.mean_top_probability or 0.0,
        },
        success=success,
        hit_step_limit=outcome.hit_step_limit,
        error=outcome.error,
    )


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def summarize(records: Sequence[Mapping[str, Any]], *, price_per_million_input_tokens: float | None) -> dict[str, Any]:
    """Run-level metrics from per-episode records (see ``cli.case_record``)."""
    n = len(records)
    successes = [r for r in records if r["success"]]
    by_type: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        by_type[r["task_type"]].append(bool(r["success"]))
    latencies = [x for r in records for x in r["decision_latencies_s"]]
    total_steps = sum(r["steps"] for r in records)
    input_tokens = sum(r["decision_input_tokens"] for r in records)
    errors: dict[str, int] = defaultdict(int)
    for r in records:
        if r["error"]:
            errors[r["error"].split(":", 1)[0]] += 1
    return {
        "episodes": n,
        "success_rate": len(successes) / n if n else 0.0,
        "success_rate_by_task_type": {
            t: {"episodes": len(v), "success_rate": sum(v) / len(v)} for t, v in sorted(by_type.items())
        },
        "mean_steps_on_success": statistics.mean(r["steps"] for r in successes) if successes else None,
        "decision_calls_per_episode": sum(r["decision_calls"] for r in records) / n if n else 0.0,
        "repeat_rate": sum(r["repeats"] for r in records) / total_steps if total_steps else 0.0,
        "hit_step_limit": sum(1 for r in records if r["hit_step_limit"]),
        "decision_latency_p50_s": _percentile(latencies, 0.5),
        "decision_latency_p95_s": _percentile(latencies, 0.95),
        "decision_input_tokens_per_episode": input_tokens / n if n else 0.0,
        "cost_per_episode_usd": (
            input_tokens * price_per_million_input_tokens / 1e6 / n
            if n and price_per_million_input_tokens is not None
            else None
        ),
        "failed": sum(errors.values()),
        "failed_by_status": dict(sorted(errors.items())),
    }
