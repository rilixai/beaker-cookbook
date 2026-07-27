"""Harvey LAB dataset loading.

The benchmark is Harvey's public Legal Agent Benchmark, laid out in the
``harveyai/harvey-labs`` GitHub repo under ``tasks/``. A task directory holds:

* ``task.json`` — ``title``, ``work_type``, ``tags``, ``instructions``,
  ``deliverables`` (a ``{filename: canonical_name}`` map the rubric grades),
  and ``criteria`` (a list of ``{id, title, deliverables, match_criteria}``
  binary pass/fail rubric items — ~60 per task on average).
* ``documents/`` — the source record (contracts, memos, spreadsheets, emails).

Task directories are **not** all one level under a practice area: larger
areas nest sub-categories (e.g. ``contracts/banking/<slug>/task.json``), so
tasks are discovered by walking for ``task.json`` recursively. A task's
``task_id`` is its directory path relative to ``tasks/`` (e.g.
``contracts/banking/<slug>``); its practice area is the first path segment.

:func:`load_records` reads task directories into normalized
:class:`HarveyLabRecord` objects. No golden reference is materialized: the
rubric ``match_criteria`` strings are the grading standard (an LLM judge
scores each criterion pass/fail). :func:`read_split` returns the frozen
train / val / test task-id lists in ``splits/`` (see ``splits/README.md``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# The frozen split lists live at the recipe root, next to this package.
SPLITS_DIR = Path(__file__).resolve().parent.parent / "splits"


@dataclass(frozen=True)
class RubricCriterion:
    """One binary-graded LAB rubric criterion.

    ``deliverables`` lists the output filenames this criterion is scoped to —
    the judge only sees those files (deliverable-scoped grading), which keeps
    a ~60-criterion task's judging cost bounded. ``match_criteria`` is the
    free-text PASS/FAIL standard the judge scores against.
    """

    id: str
    title: str
    match_criteria: str
    deliverables: tuple[str, ...]


@dataclass(frozen=True)
class HarveyLabRecord:
    """Normalized Harvey LAB task record used by the agent + judge."""

    task_id: str
    practice_area: str
    title: str
    work_type: str
    instructions: str
    deliverables: Mapping[str, str]
    criteria: tuple[RubricCriterion, ...]
    documents: tuple[str, ...]
    raw_task: Mapping[str, Any]

    @property
    def deliverable_names(self) -> tuple[str, ...]:
        """Output filenames the agent must produce (rubric-graded)."""
        return tuple(self.deliverables.keys())


def _coerce_criteria(value: Any) -> tuple[RubricCriterion, ...]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return ()
    if not isinstance(value, (list, tuple)):
        return ()
    out: list[RubricCriterion] = []
    for idx, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            continue
        match = str(entry.get("match_criteria") or "").strip()
        if not match:
            continue
        deliverables = tuple(str(d) for d in (entry.get("deliverables") or ()) if d)
        out.append(
            RubricCriterion(
                id=str(entry.get("id") or f"C-{idx + 1:03d}"),
                title=str(entry.get("title") or ""),
                match_criteria=match,
                deliverables=deliverables,
            )
        )
    return tuple(out)


def _coerce_deliverables(value: Any) -> dict[str, str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return {}
    if isinstance(value, Mapping):
        return {str(k): str(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return {str(v): str(v) for v in value if v}
    return {}


def _load_task_dir(task_dir: Path, task_id: str) -> HarveyLabRecord:
    raw = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    docs_dir = task_dir / "documents"
    documents = sorted(p.relative_to(docs_dir).as_posix() for p in docs_dir.rglob("*") if p.is_file())
    tid = task_id
    return HarveyLabRecord(
        task_id=tid,
        practice_area=tid.split("/", 1)[0],
        title=str(raw.get("title") or ""),
        work_type=str(raw.get("work_type") or ""),
        instructions=str(raw.get("instructions") or ""),
        deliverables=_coerce_deliverables(raw.get("deliverables")),
        criteria=_coerce_criteria(raw.get("criteria")),
        documents=tuple(documents),
        raw_task=dict(raw),
    )


def load_records(
    tasks_root: str | Path,
    *,
    task_ids: Sequence[str] | None = None,
) -> list[HarveyLabRecord]:
    """Load + normalize LAB task records from a local ``harvey-labs`` checkout.

    ``tasks_root`` is the repo's ``tasks/`` directory. Tasks are discovered by
    walking for ``task.json`` recursively (areas nest sub-categories), so a
    ``task_id`` is the task directory's path relative to ``tasks/``.

    ``task_ids`` (e.g. from :func:`read_split`) loads exactly those tasks in
    the given order — cheap for a capped run. Without it, every task is loaded
    in sorted ``task_id`` order.
    """
    base = Path(tasks_root)
    if not base.is_dir():
        raise FileNotFoundError(f"tasks_root {base} is not a directory.")
    if task_ids is not None:
        records: list[HarveyLabRecord] = []
        missing: list[str] = []
        for tid in task_ids:
            task_dir = base / tid
            if not (task_dir / "task.json").is_file():
                missing.append(tid)
                continue
            records.append(_load_task_dir(task_dir, tid))
        # Fail loudly: a frozen split id absent from the tree means the checkout
        # doesn't match the pinned commit, and silently shrinking the run would
        # break reproducibility and make scores non-comparable.
        if missing:
            preview = ", ".join(missing[:10]) + (" …" if len(missing) > 10 else "")
            raise FileNotFoundError(
                f"{len(missing)} split task id(s) have no task.json under {base} "
                f"(is the checkout at the pinned HARVEY_LABS_COMMIT?): {preview}"
            )
        return records
    # ``as_posix`` so discovered task IDs use "/" on every OS and match the
    # committed ``splits/*.txt`` lists (and the "/"-based practice-area parse).
    return [
        _load_task_dir(tj.parent, tj.parent.relative_to(base).as_posix()) for tj in sorted(base.rglob("task.json"))
    ]


def read_split(split: str) -> list[str]:
    """Return the frozen task-id list for ``split`` in {train, val, test}.

    Lists are stored round-robin across practice areas, so any prefix stays
    distribution-representative — a run cap can take the first N ids.
    """
    path = SPLITS_DIR / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"No split file {path} (expected one of train/val/test).")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


__all__ = [
    "HarveyLabRecord",
    "RubricCriterion",
    "SPLITS_DIR",
    "load_records",
    "read_split",
]
