"""Batch runner: answer every case, then grade every rubric criterion.

``run_records`` only calls the agent. ``evaluate_records`` reuses any
responses already in ``run_outputs.json`` (keyed by ``prompt_id``), answers
the rest, then grades. One case failing never aborts the batch: it is kept
with its error and scores 0.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from ..agent.agent import HealthBenchAgent
from ..data.dataset import HealthBenchRecord
from .grader import RubricGrader
from .scoring import OVERALL, aggregate, case_metrics


logger = logging.getLogger("healthbench")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


async def run_records(
    agent: HealthBenchAgent, records: list[HealthBenchRecord], *, max_concurrency: int
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _one(record: HealthBenchRecord) -> dict[str, Any]:
        async with semaphore:
            try:
                output = await agent.respond(record)
            except Exception as exc:  # noqa: BLE001 - report, don't abort
                logger.warning("Case %s failed: %s", record.prompt_id, exc)
                return {"prompt_id": record.prompt_id, "error": f"{type(exc).__name__}: {exc}"}
        return {"prompt_id": record.prompt_id, "response": output.response, "usage": output.usage}

    return list(await asyncio.gather(*(_one(r) for r in records)))


async def evaluate_records(
    agent: HealthBenchAgent,
    grader: RubricGrader,
    records: list[HealthBenchRecord],
    *,
    max_concurrency: int,
    previous: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return ``(run_outputs, per_case, summary)``."""
    previous = previous or {}
    todo = [r for r in records if "response" not in previous.get(r.prompt_id, {})]
    if len(todo) < len(records):
        logger.info("Reusing %d saved response(s); answering %d.", len(records) - len(todo), len(todo))
    fresh = {o["prompt_id"]: o for o in await run_records(agent, todo, max_concurrency=max_concurrency)}
    outputs = [fresh.get(r.prompt_id) or previous[r.prompt_id] for r in records]

    async def _grade(record: HealthBenchRecord, output: dict[str, Any]) -> dict[str, Any]:
        base = {"prompt_id": record.prompt_id, "example_tags": record.example_tags}
        if "error" in output:
            return {**base, "error": output["error"], "metrics": {OVERALL: 0.0}}
        try:
            grades = await grader.grade(record.prompt, output["response"], record.rubrics)
        except Exception as exc:  # noqa: BLE001 - report, don't abort
            logger.warning("Grading %s failed: %s", record.prompt_id, exc)
            return {**base, "error": f"grader: {type(exc).__name__}: {exc}", "metrics": {OVERALL: 0.0}}
        return {
            **base,
            "response": output["response"],
            "metrics": case_metrics(grades, record.example_tags),
            "grades": [g.__dict__ for g in grades],
        }

    per_case = list(await asyncio.gather(*(_grade(r, o) for r, o in zip(records, outputs, strict=True))))
    summary = {
        "num_cases": len(per_case),
        "num_errored": sum(1 for c in per_case if "error" in c),
        "metrics": aggregate([c["metrics"] for c in per_case]),
    }
    return outputs, per_case, summary
