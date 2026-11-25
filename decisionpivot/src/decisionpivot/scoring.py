"""Score one contrast group, and score a run with the pinned upstream ``scoring/score.py``. Fixed.

Per item (as upstream ``score.py``): the top option is the option with the
largest probability, the first listed winning a tie; an item is right when its
top option is the gold option. A failed item counts as wrong with the uniform
distribution.

Per group: ``objective`` = pair correct, 1 only if every member is right.

Run-level numbers are not computed here. ``write_run`` writes upstream answer
files and a scoring folder, and ``upstream_scores`` runs upstream ``score.py``
on them unmodified and returns its numbers.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decisionpivot.data import FAMILIES, Row


SYSTEM = "decisionpivot-agent"


def top_option(keys: Sequence[str], probabilities: Mapping[str, float] | None) -> tuple[str | None, float]:
    """Upstream's top option and top probability; for a failed item, no option and the uniform probability."""
    if probabilities is None:
        return None, 1.0 / len(keys)
    best = max(keys, key=lambda k: probabilities[k])  # max keeps the first of equal values
    return best, probabilities[best]


@dataclass(frozen=True)
class ItemScore:
    id: str
    expected: str
    predicted: str | None
    correct: bool
    top_probability: float
    error: str | None


@dataclass(frozen=True)
class GroupScore:
    objective: float
    field_scores: dict[str, float]
    items: list[ItemScore]


def _item_outputs(output: Any) -> dict[str, Mapping[str, Any]]:
    items = output.get("items") if isinstance(output, Mapping) else None
    if not isinstance(items, list):
        return {}
    return {str(o.get("id")): o for o in items if isinstance(o, Mapping)}


def score_group(row: Row, output: Any) -> GroupScore:
    outputs = _item_outputs(output)
    scores = []
    for item in row.input.items:
        keys = item.question.keys
        out = outputs.get(item.id)
        error = "no output" if out is None else out.get("error")
        probabilities = out.get("probabilities") if out is not None and error is None else None
        if error is None and (not isinstance(probabilities, Mapping) or set(probabilities) != set(keys)):
            error, probabilities = "malformed_answer: probabilities do not cover the options", None
        predicted, top_p = top_option(keys, probabilities)
        gold = row.expected.gold[item.id]
        message = None if error is None else str(error)
        scores.append(ItemScore(item.id, gold, predicted, predicted == gold, top_p, message))
    right = sum(s.correct for s in scores)
    pair = float(right == len(scores))
    fields = {
        "pair_correct": pair,
        "item_accuracy": right / len(scores),
        "min_top_probability": min(s.top_probability for s in scores),
    }
    if row.expected.level == 3:  # its mean is level-3 pair accuracy; field scores must lie in [0, 1]
        fields["level3_pair_correct"] = pair
    return GroupScore(
        objective=pair,
        field_scores=fields,
        items=scores,
    )


def write_run(
    run_root: Path, rows: Sequence[Row], outputs: Mapping[str, Mapping[str, Any]], upstream: Path
) -> tuple[Path, Path]:
    """Upstream answer files (``runs/<system>/<family>.jsonl``) and the matching scoring folder.

    The scoring folder holds, per family, exactly the items that were run, in
    upstream's item format without the state (the scorer does not read it), and
    upstream's release 1.1 task card.
    """
    bench, runs = run_root / "bench", run_root / "runs"
    system_dir = runs / SYSTEM
    system_dir.mkdir(parents=True, exist_ok=True)
    by_family: dict[str, list[Row]] = {}
    for row in rows:
        by_family.setdefault(row.input.family, []).append(row)
    for family, members in by_family.items():
        task_dir = bench / "tasks" / family
        task_dir.mkdir(parents=True, exist_ok=True)
        card = json.loads((upstream / "releases/v1.1/task_cards" / f"{family}.json").read_text())
        (task_dir / "task.json").write_text(json.dumps(card, indent=1))
        items, answers = [], []
        for row in members:
            per_item = _item_outputs(outputs.get(row.id, {}))
            for item in row.input.items:
                items.append(
                    {
                        "id": item.id,
                        "state": None,
                        "questions": {"q": item.question.model_dump(mode="json")},
                        "gold": row.expected.gold[item.id],
                        "soft": None,
                        "group": row.group,
                        "meta": {"level": row.expected.level},
                    }
                )
                out = per_item.get(item.id)
                if out is None or out.get("error") is not None or out.get("probabilities") is None:
                    answers.append({"id": item.id, "error": str((out or {}).get("error") or "no output")})
                else:
                    answers.append({"id": item.id, "p": out["probabilities"]})
        (task_dir / "items.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
        (system_dir / f"{family}.jsonl").write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in answers))
    (run_root / "systems.json").write_text(json.dumps({SYSTEM: {"name": SYSTEM, "table": True}}))
    return bench, runs


def upstream_scores(
    run_root: Path, upstream: Path, families: Sequence[str], *, resamples: int = 1000
) -> dict[str, Any]:
    """Run the pinned upstream ``score.py --all`` and return its scores for our system."""
    out = run_root / "scores"
    families = [f for f in FAMILIES if f in families]
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(upstream / "scoring/score.py"),
            "--all",
            "--root",
            str(run_root / "bench"),
            "--runs",
            str(run_root / "runs"),
            "--systems",
            str(run_root / "systems.json"),
            "--headline-families",
            ",".join(families),
            "--resamples",
            str(resamples),
            "--out",
            str(out),
        ],
        cwd=upstream,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Upstream score.py failed:\n{result.stdout}\n{result.stderr}")
    system = json.loads((out / "scores.json").read_text())["systems"][SYSTEM]
    if system["not_run"]:
        raise RuntimeError(f"Upstream score.py did not count these families: {system['not_run']}")
    return {"core": system["core"], "tasks": system["tasks"], "report": str(out / "report.md")}


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def latency_and_cost(
    item_outputs: Sequence[Mapping[str, Any]], *, price_per_million_input_tokens: float | None
) -> dict[str, Any]:
    """p50/p95 latency per decision call, and cost per 1,000 decisions (input tokens only, as Jev bills)."""
    latencies = [x for o in item_outputs for x in o.get("decision_latency_s", [])]
    calls = sum(o.get("decision_calls", 0) for o in item_outputs)
    tokens = sum(o.get("decision_input_tokens", 0) for o in item_outputs)
    cost = (
        None
        if not calls or price_per_million_input_tokens is None
        else tokens / calls * 1000 * price_per_million_input_tokens / 1e6
    )
    return {
        "decisions": calls,
        "latency_per_decision_s": {"p50": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95)},
        "input_tokens_per_decision": tokens / calls if calls else None,
        "cost_per_1000_decisions_usd": cost,
    }
