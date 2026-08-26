"""Task loading and frozen-split resolution.

Samples come straight from ``automationbench.domains`` (in-process Python task
builders — no network). The split files key tasks by ``task_name``
(``<domain>.<name>``), the benchmark's only globally-unique stable id: the
dataset's ``example_id`` column is only unique *within* a domain (e.g. sales
and marketing both start at different offsets but collide across the combined
600-task set), so it cannot key a cross-domain split. verifiers assigns its
own positional integer ``example_id`` at rollout time; we carry the sample's
global index for that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any


PUBLIC_DOMAINS = ["sales", "marketing", "operations", "support", "finance", "hr"]
SPLITS_DIR = Path(__file__).parent.parent / "splits"


@dataclass(frozen=True)
class Sample:
    """One AutomationBench task, ready to run as a single verifiers rollout."""

    task_name: str  # globally-unique stable id, e.g. "sales.multi_hop_lookup"
    domain: str
    index: int  # global position in the combined public dataset (verifiers example_id)
    prompt: list[dict[str, Any]]
    answer: str
    info: dict[str, Any]  # zapier_tools / initial_state / assertions / task_name


def task_family(task_name: str) -> str:
    """Task family used for split stratification: the first underscore token of
    the name part (e.g. ``sales.docusign_contract_send`` -> ``docusign``)."""
    name = task_name.split(".", 1)[1] if "." in task_name else task_name
    return name.split("_", 1)[0]


@cache
def load_samples(include_simple: bool = False) -> tuple[Sample, ...]:
    """Build every public task (plus optionally the unscored ``simple`` domain)
    in the benchmark's own deterministic order."""
    from automationbench.domains import get_combined_dataset

    domains = PUBLIC_DOMAINS + (["simple"] if include_simple else [])
    dataset = get_combined_dataset(domains)
    samples: list[Sample] = []
    for index, row in enumerate(dataset):
        info = row["info"]
        if isinstance(info, str):
            info = json.loads(info)
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
    """Load the frozen ``train``/``test`` split (or the optional unscored
    ``simple`` extra-training list) as Samples, in split-file order."""
    names = read_split_names(split)
    by_name = {s.task_name: s for s in load_samples(include_simple=split == "simple")}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(
            f"Split {split!r} names {len(missing)} task(s) absent from the pinned "
            f"automation-bench task set (first: {missing[0]!r}). The dependency pin "
            "and the frozen split files must move together."
        )
    return [by_name[n] for n in names]
