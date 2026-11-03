"""Task loading and frozen-split resolution.

Samples come straight from ``automationbench.domains`` (in-process Python task
builders — no network). The split files key tasks by ``task_name``
(``<domain>.<name>``), the benchmark's only globally-unique stable id: the
dataset's numeric ``example_id`` values collide across domains. The recipe
assigns a global positional index for each rollout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any


PUBLIC_DOMAINS = ["sales", "marketing", "operations", "support", "finance", "hr", "simple"]
SPLITS_DIR = Path(__file__).parent.parent / "splits"


@dataclass(frozen=True)
class Sample:
    """One AutomationBench task, ready to run as a single PI rollout."""

    task_name: str  # globally-unique stable id, e.g. "sales.multi_hop_lookup"
    domain: str
    index: int  # global position in the combined public dataset (rollout example_id)
    prompt: list[dict[str, Any]]
    answer: str
    info: dict[str, Any]  # zapier_tools / initial_state / assertions / task_name


def task_family(task_name: str) -> str:
    """Task family used for split stratification: the first underscore token of
    the name part (e.g. ``sales.docusign_contract_send`` -> ``docusign``)."""
    name = task_name.split(".", 1)[1] if "." in task_name else task_name
    return name.split("_", 1)[0]


@cache
def load_samples() -> tuple[Sample, ...]:
    """Build all 800 published PI tasks in the benchmark's own deterministic order."""
    from automationbench.domains import get_combined_dataset
    from automationbench.schema.world import WorldMeta

    dataset = get_combined_dataset(PUBLIC_DOMAINS)
    samples: list[Sample] = []
    for index, row in enumerate(dataset):
        info = row["info"]
        if isinstance(info, str):
            info = json.loads(info)
        meta = info.setdefault("initial_state", {}).setdefault("meta", {})
        if not meta.get("current_time"):
            meta["current_time"] = WorldMeta().current_time.isoformat()
        task_name = info["task_name"]
        samples.append(
            Sample(
                task_name=task_name,
                domain=task_name.split(".", 1)[0],
                index=index,
                prompt=row["prompt"],
                answer=row.get("answer", "") or "",
                info=info,
            )
        )
    return tuple(samples)


def split_path(split: str) -> Path:
    return SPLITS_DIR / f"{split}.txt"


def read_split_names(split: str) -> list[str]:
    path = split_path(split)
    if not path.is_file():
        raise FileNotFoundError(f"Unknown split {split!r} (no {path})")
    names = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(line)
    return names


def load_split(split: str) -> list[Sample]:
    """Load eligible train/test cases in frozen split order."""
    names = read_split_names(split)
    if split in {"train", "test"}:
        excluded = set(read_split_names("excluded_cases"))
        names = [name for name in names if name not in excluded]
    by_name = {s.task_name: s for s in load_samples()}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(
            f"Split {split!r} names {len(missing)} task(s) absent from the pinned "
            f"automation-bench task set (first: {missing[0]!r}). The dependency pin "
            "and the frozen split files must move together."
        )
    return [by_name[n] for n in names]
