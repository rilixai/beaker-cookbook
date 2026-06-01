"""APEX-Agents dataset loading and conversion to optimizer cases.

The benchmark is hosted on the HuggingFace dataset ``mercor/apex-agents``.
We focus on the **investment-banking** subset (``domain ==
"Investment Banking"``) — 160 long-horizon professional-knowledge-work
tasks across 10 worlds.

Three HF dataset files back the loader:

* ``tasks_and_rubrics.json`` — a JSON array of task objects. Each task
  carries ``task_id``, ``task_name``, ``domain``, ``prompt``,
  ``world_id``, an optional ``task_input_files`` list, and a ``rubric``
  (a list of criteria objects; the first entry is the primary
  objective). Each criterion has ``verifier_id`` + ``criteria``.
* ``world_descriptions.json`` — a JSON array of world objects
  (``world_id``, ``world_name`` + a free-form description).
* ``world_files_zipped/{world_id}.zip`` — the world's file assets.
* ``task_files/{task_id}/**`` — optional per-task input files.

``huggingface_hub`` is imported lazily so this module stays importable
/ testable offline — tests monkeypatch ``sys.modules`` with a stub
loader per the SWE-bench / IFBench pattern.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rilixai.prompt_optimization.models import Sample


logger = logging.getLogger(__name__)


# Ground-truth key the APEX-Agents metrics calculator points its single
# field config at. The per-case comparator reads back the precomputed
# ``rubric_pass_rate`` float (the runtime ran the LLM judge), but we
# still bundle the rubric + prompt + world/task ids here so the
# feedback strings / judge have access to the criteria.
_APEX_AGENTS_GROUND_TRUTH_KEY = "__apex_agents_rubric__"


# The investment-banking subset is the one we calibrate to. Override
# via ``domain`` on the loaders for other professional-knowledge-work
# subsets.
DEFAULT_DOMAIN = "Investment Banking"


# HF dataset coordinates.
APEX_AGENTS_HF_REPO = "mercor/apex-agents"
_TASKS_FILE = "tasks_and_rubrics.json"
_WORLDS_FILE = "world_descriptions.json"


@dataclass(frozen=True)
class RubricCriterion:
    """One binary-graded rubric criterion.

    ``verifier_id`` selects the verifier Mercor's Archipelago harness
    uses (default ``output_llm`` — an LLM judge). ``criteria`` is the
    free-text statement the judge scores Met / Not-met.
    """

    verifier_id: str
    criteria: str


@dataclass(frozen=True)
class ApexAgentsRecord:
    """Normalized APEX-Agents record used by the runtime and metrics.

    ``raw_task`` is the full HF task dict — downstream code uses it for
    debugging / feedback. ``rubric`` is the ordered list of criteria;
    the first entry is the primary objective.
    """

    task_id: str
    task_name: str
    domain: str
    prompt: str
    world_id: str
    rubric: tuple[RubricCriterion, ...]
    task_input_files: tuple[str, ...]
    raw_task: Mapping[str, Any]
    world_name: str = ""
    world_description: str = ""


def _coerce_rubric(value: Any) -> tuple[RubricCriterion, ...]:
    """Coerce the HF ``rubric`` field into a tuple of :class:`RubricCriterion`.

    Accepts the canonical list-of-dicts form. JSON-string-encoded
    rubrics (some HF exports stringify nested structures) are decoded
    first. Anything else yields an empty tuple — a task without a
    rubric is unscoreable, which the metrics layer treats as skipped.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        try:
            value = json.loads(text)
        except Exception:
            return ()
    if not isinstance(value, (list, tuple)):
        return ()
    out: list[RubricCriterion] = []
    for entry in value:
        if not isinstance(entry, Mapping):
            continue
        criteria = str(entry.get("criteria") or "").strip()
        if not criteria:
            continue
        verifier_id = str(entry.get("verifier_id") or "output_llm").strip() or "output_llm"
        out.append(RubricCriterion(verifier_id=verifier_id, criteria=criteria))
    return tuple(out)


def _coerce_str_list(value: Any) -> tuple[str, ...]:
    """Coerce an optional list-of-strings HF field into a tuple of str."""
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        try:
            parsed = json.loads(text)
        except Exception:
            return (text,)
        if isinstance(parsed, (list, tuple)):
            return tuple(str(v) for v in parsed if v)
        return (str(parsed),)
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value if v)
    return ()


def _normalize_record(
    raw: Mapping[str, Any],
    *,
    index_hint: int,
    worlds: Mapping[str, Mapping[str, Any]] | None = None,
) -> ApexAgentsRecord:
    tid_raw = raw.get("task_id")
    task_id = str(tid_raw) if tid_raw is not None else f"apex-agents-{index_hint}"
    task_name = str(raw.get("task_name") or "")
    domain = str(raw.get("domain") or "")
    prompt = str(raw.get("prompt") or "")
    world_id = str(raw.get("world_id") or "")
    rubric = _coerce_rubric(raw.get("rubric"))
    task_input_files = _coerce_str_list(raw.get("task_input_files"))
    world_name = ""
    world_description = ""
    if worlds is not None:
        world_meta = worlds.get(world_id)
        if isinstance(world_meta, Mapping):
            world_name = str(world_meta.get("world_name") or "")
            world_description = str(
                world_meta.get("description") or world_meta.get("world_description") or world_meta.get("text") or ""
            )
    return ApexAgentsRecord(
        task_id=task_id,
        task_name=task_name,
        domain=domain,
        prompt=prompt,
        world_id=world_id,
        rubric=rubric,
        task_input_files=task_input_files,
        raw_task=dict(raw),
        world_name=world_name,
        world_description=world_description,
    )


def record_to_case(record: ApexAgentsRecord, *, group_key: str | None = None) -> Sample:
    """Convert a normalized APEX-Agents record into a rilixai :class:`Sample`.

    The ground-truth bundle stashes the rubric + prompt + world/task
    ids the runtime needs to run the LLM judge after the agent
    finishes. Group key defaults to ``world_id`` so failure-focused
    samplers (and the world-level k-fold splitter) can stratify by
    world.
    """
    resolved_group = group_key or (record.world_id or "apex_agents")
    rubric_payload = [{"verifier_id": c.verifier_id, "criteria": c.criteria} for c in record.rubric]
    bundle = {
        "task_id": record.task_id,
        "world_id": record.world_id,
        "prompt": record.prompt,
        "rubric": rubric_payload,
    }
    ground_truth = {
        "task_id": record.task_id,
        "world_id": record.world_id,
        "prompt": record.prompt,
        "rubric": rubric_payload,
        _APEX_AGENTS_GROUND_TRUTH_KEY: bundle,
    }
    return Sample(
        input=record,
        sample_id=record.task_id,
        ground_truth=ground_truth,
        group_key=resolved_group,
        metadata={
            "domain": record.domain,
            "world_id": record.world_id,
            "world_name": record.world_name,
            "task_name": record.task_name,
            "num_rubric_criteria": len(record.rubric),
        },
    )


def cases_from_records(
    records: Iterable[Mapping[str, Any] | ApexAgentsRecord],
    *,
    group_key: str | None = None,
) -> list[Sample]:
    """Build a list of optimizer cases from raw or normalized records."""
    cases: list[Sample] = []
    for idx, raw in enumerate(records):
        if isinstance(raw, ApexAgentsRecord):
            record = raw
        elif isinstance(raw, Mapping):
            record = _normalize_record(raw, index_hint=idx)
        else:
            raise TypeError(
                f"cases_from_records expected Mapping or ApexAgentsRecord at position {idx}, got {type(raw).__name__}."
            )
        cases.append(record_to_case(record, group_key=group_key))
    return cases


def filter_investment_banking(
    records: Sequence[ApexAgentsRecord],
    *,
    domain: str = DEFAULT_DOMAIN,
) -> list[ApexAgentsRecord]:
    """Return only the records whose ``domain`` matches (case-insensitive).

    Underscores are normalized to spaces so CLI/sandbox tokens like
    ``"investment_banking"`` match the dataset's ``"Investment Banking"``
    (the choice values are slug-style; the HF ``domain`` field is title-cased
    with a space). Without this, ``--domain investment_banking`` matched zero
    records and silently produced empty train/val splits.
    """
    target = domain.strip().casefold().replace("_", " ")
    return [r for r in records if r.domain.strip().casefold().replace("_", " ") == target]


def _hf_download(repo_id: str, filename: str, *, cache_dir: str | None) -> str:
    """Download one file from a HF dataset repo, returning the local path.

    Pushed into a helper so the test suite can monkeypatch a single
    place (an :class:`ImportError` if ``huggingface_hub`` is missing).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Loading the APEX-Agents dataset from HuggingFace requires the optional "
            "`huggingface-hub` dependency. Install it with `pip install huggingface-hub`."
        ) from exc
    return str(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
    )


def _read_json_array(path: str) -> list[Mapping[str, Any]]:
    """Read a JSON file expected to hold a top-level array of objects."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array at {path}, got {type(data).__name__}.")
    return [row for row in data if isinstance(row, Mapping)]


def _worlds_by_id(world_rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in world_rows:
        wid = str(row.get("world_id") or "")
        if wid:
            out[wid] = row
    return out


def load_apex_agents_records(
    *,
    domain: str | None = DEFAULT_DOMAIN,
    max_records: int | None = None,
    cache_dir: str | None = None,
    repo_id: str = APEX_AGENTS_HF_REPO,
) -> list[ApexAgentsRecord]:
    """Load + normalize APEX-Agents task records from HuggingFace.

    1. Download ``tasks_and_rubrics.json`` + ``world_descriptions.json``.
    2. Normalize every task row (attaching world name/description).
    3. Filter to ``domain`` (default ``"Investment Banking"``; pass
       ``None`` to keep every domain).
    4. Optionally cap to ``max_records`` from the front of the
       (filtered) list. Records are returned in file order — no shuffle
       — so the world-level k-fold splitter is the single source of
       randomness.
    """
    if max_records is not None and max_records < 1:
        raise ValueError(f"load_apex_agents_records max_records must be >= 1 or None, got {max_records}.")
    tasks_path = _hf_download(repo_id, _TASKS_FILE, cache_dir=cache_dir)
    worlds_path = _hf_download(repo_id, _WORLDS_FILE, cache_dir=cache_dir)
    task_rows = _read_json_array(tasks_path)
    world_rows = _read_json_array(worlds_path)
    worlds = _worlds_by_id(world_rows)
    records = [_normalize_record(row, index_hint=i, worlds=worlds) for i, row in enumerate(task_rows)]
    if domain is not None:
        records = filter_investment_banking(records, domain=domain)
    if max_records is not None:
        records = records[:max_records]
    return records


def load_apex_agents_cases(
    *,
    domain: str | None = DEFAULT_DOMAIN,
    max_records: int | None = None,
    cache_dir: str | None = None,
    repo_id: str = APEX_AGENTS_HF_REPO,
) -> list[Sample]:
    """Load the APEX-Agents records and convert them to optimizer cases."""
    records = load_apex_agents_records(
        domain=domain,
        max_records=max_records,
        cache_dir=cache_dir,
        repo_id=repo_id,
    )
    return cases_from_records(records)


def world_ids_for_cases(cases: Iterable[Sample]) -> list[str]:
    """Return the sorted distinct ``world_id`` group keys across cases."""
    seen: set[str] = set()
    for case in cases:
        wid = str(case.metadata.get("world_id") or case.group_key or "")
        if wid:
            seen.add(wid)
    return sorted(seen)


__all__ = [
    "APEX_AGENTS_HF_REPO",
    "DEFAULT_DOMAIN",
    "ApexAgentsRecord",
    "RubricCriterion",
    "_APEX_AGENTS_GROUND_TRUTH_KEY",
    "cases_from_records",
    "filter_investment_banking",
    "load_apex_agents_cases",
    "load_apex_agents_records",
    "record_to_case",
    "world_ids_for_cases",
]
