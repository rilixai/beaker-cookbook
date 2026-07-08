"""Harvey LAB dataset loading and conversion to optimizer cases.

The benchmark is Harvey's public Legal Agent Benchmark, laid out in the
``harveyai/harvey-labs`` GitHub repo as ``tasks/<practice_area>/<slug>/``
directories. Each task directory holds:

* ``task.json`` — ``title``, ``work_type``, ``tags``, ``instructions``,
  ``deliverables`` (a ``{filename: canonical_name}`` map the rubric grades),
  and ``criteria`` (a list of ``{id, title, deliverables, match_criteria}``
  binary pass/fail rubric items — ~60 per task on average).
* ``documents/`` — the source record (contracts, memos, spreadsheets, emails).

Two loading paths share one normalizer:

* :func:`load_harvey_lab_records` reads task directories from a local
  checkout (used by the export script + tests).
* :class:`HarveyLabDataLoader` parses the exported JSONL rows a hosted
  ``rilixai run`` streams in (the task documents are fetched at runtime by
  the workspace's GitHub task source, keyed by ``task_id`` + ``documents``).

No golden reference is materialized: the rubric ``match_criteria`` strings
are the grading standard (an LLM judge scores each criterion pass/fail).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rilixai import Case, CaseDataLoader, DatasetRowContext, DatasetSchema


logger = logging.getLogger(__name__)


# Ground-truth key the scorer reads the rubric bundle back from.
_HARVEY_LAB_GROUND_TRUTH_KEY = "__harvey_lab_rubric__"


HARVEY_LAB_DATASET_SCHEMA = DatasetSchema(
    json_schema={
        "type": "object",
        "required": ["task_id", "practice_area", "instructions", "deliverables", "criteria"],
        "properties": {
            "task_id": {"type": "string", "minLength": 1},
            "practice_area": {"type": "string", "minLength": 1},
            "title": {"type": "string"},
            "work_type": {"type": "string"},
            "instructions": {"type": "string"},
            "deliverables": {"type": "object"},
            "documents": {"type": "array", "items": {"type": "string"}},
            "criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["match_criteria"],
                    "properties": {
                        "id": {"type": "string"},
                        "title": {"type": "string"},
                        "match_criteria": {"type": "string"},
                        "deliverables": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        },
        "additionalProperties": True,
    },
)


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
    """Normalized Harvey LAB task record used by the agent + scorer."""

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


def _normalize_record(raw: Mapping[str, Any], *, index_hint: int) -> HarveyLabRecord:
    tid = str(raw.get("task_id") or f"harvey-lab-{index_hint}")
    practice_area = str(raw.get("practice_area") or (tid.split("/", 1)[0] if "/" in tid else "harvey_lab"))
    return HarveyLabRecord(
        task_id=tid,
        practice_area=practice_area,
        title=str(raw.get("title") or ""),
        work_type=str(raw.get("work_type") or ""),
        instructions=str(raw.get("instructions") or ""),
        deliverables=_coerce_deliverables(raw.get("deliverables")),
        criteria=_coerce_criteria(raw.get("criteria")),
        documents=tuple(str(d) for d in (raw.get("documents") or ()) if d),
        raw_task=dict(raw),
    )


def record_to_row(record: HarveyLabRecord) -> dict[str, Any]:
    """Serialize a record to the JSONL row shape the DataLoader parses."""
    return {
        "task_id": record.task_id,
        "practice_area": record.practice_area,
        "title": record.title,
        "work_type": record.work_type,
        "instructions": record.instructions,
        "deliverables": dict(record.deliverables),
        "documents": list(record.documents),
        "criteria": [
            {
                "id": c.id,
                "title": c.title,
                "match_criteria": c.match_criteria,
                "deliverables": list(c.deliverables),
            }
            for c in record.criteria
        ],
    }


def record_to_case(record: HarveyLabRecord, *, group_key: str | None = None) -> Case:
    """Convert a normalized record into a rilixai :class:`Case`.

    Group key defaults to ``practice_area`` so splits stratify by practice
    area (the leakage-free axis: a GEPA-optimized prompt is validated on
    practice areas it did not train on). The ground-truth bundle carries the
    rubric criteria + deliverables map + task metadata the runtime's judge
    needs after the agent finishes.
    """
    resolved_group = group_key or (record.practice_area or "harvey_lab")
    criteria_payload = [
        {"id": c.id, "title": c.title, "match_criteria": c.match_criteria, "deliverables": list(c.deliverables)}
        for c in record.criteria
    ]
    bundle = {
        "task_id": record.task_id,
        "title": record.title,
        "instructions": record.instructions,
        "deliverables": dict(record.deliverables),
        "criteria": criteria_payload,
    }
    ground_truth = {**bundle, _HARVEY_LAB_GROUND_TRUTH_KEY: bundle}
    return Case(
        input=record,
        case_id=record.task_id,
        ground_truth=ground_truth,
        group_key=resolved_group,
        metadata={
            "practice_area": record.practice_area,
            "title": record.title,
            "work_type": record.work_type,
            "num_criteria": len(record.criteria),
            "num_documents": len(record.documents),
        },
    )


def cases_from_records(
    records: Iterable[Mapping[str, Any] | HarveyLabRecord],
    *,
    group_key: str | None = None,
) -> list[Case]:
    """Build optimizer cases from raw or normalized records."""
    cases: list[Case] = []
    for idx, raw in enumerate(records):
        record = raw if isinstance(raw, HarveyLabRecord) else _normalize_record(raw, index_hint=idx)
        cases.append(record_to_case(record, group_key=group_key))
    return cases


def load_harvey_lab_records(
    tasks_root: str | Path,
    *,
    practice_areas: Sequence[str] | None = None,
    max_per_area: int | None = None,
) -> list[HarveyLabRecord]:
    """Load + normalize LAB task records from a local ``harvey-labs`` checkout.

    ``tasks_root`` is the repo's ``tasks/`` directory. Walks
    ``<practice_area>/<slug>/task.json`` in stable sorted order (no shuffle —
    the splitter is the single source of randomness), attaching the sorted
    ``documents/`` filenames so the runtime task source can fetch them.
    """
    base = Path(tasks_root)
    if not base.is_dir():
        raise FileNotFoundError(f"tasks_root {base} is not a directory.")
    wanted = {p.strip() for p in practice_areas} if practice_areas else None
    records: list[HarveyLabRecord] = []
    for area_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if wanted is not None and area_dir.name not in wanted:
            continue
        count = 0
        for task_json in sorted(area_dir.glob("*/task.json")):
            if max_per_area is not None and count >= max_per_area:
                break
            raw = json.loads(task_json.read_text(encoding="utf-8"))
            task_dir = task_json.parent
            documents = sorted(
                str(p.relative_to(task_dir / "documents")) for p in (task_dir / "documents").rglob("*") if p.is_file()
            )
            merged = {
                **raw,
                "task_id": f"{area_dir.name}/{task_dir.name}",
                "practice_area": area_dir.name,
                "documents": documents,
            }
            records.append(_normalize_record(merged, index_hint=len(records)))
            count += 1
    return records


class HarveyLabDataLoader(CaseDataLoader[HarveyLabRecord]):
    """Typed rilixai data loader over exported Harvey LAB JSONL rows."""

    dataset_schema = HARVEY_LAB_DATASET_SCHEMA

    def parse_row(self, raw: Mapping[str, Any], context: DatasetRowContext) -> HarveyLabRecord:
        return _normalize_record(raw, index_hint=context.line_number or 0)

    def iter_cases(self, row: HarveyLabRecord, context: DatasetRowContext) -> Iterable[Case]:
        del context
        yield record_to_case(row)


def practice_areas_for_cases(cases: Iterable[Case]) -> list[str]:
    """Return the sorted distinct practice-area group keys across cases."""
    seen: set[str] = set()
    for case in cases:
        area = str(case.metadata.get("practice_area") or case.group_key or "")
        if area:
            seen.add(area)
    return sorted(seen)


__all__ = [
    "HARVEY_LAB_DATASET_SCHEMA",
    "HarveyLabDataLoader",
    "HarveyLabRecord",
    "RubricCriterion",
    "_HARVEY_LAB_GROUND_TRUTH_KEY",
    "cases_from_records",
    "load_harvey_lab_records",
    "practice_areas_for_cases",
    "record_to_case",
    "record_to_row",
]
