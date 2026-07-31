"""RilixAI prompt-optimization spec for the Harvey LAB legal agent.

What is optimized: the recipe's **two agent prompts** — Artificial Analysis'
LAB-AA ``system_prompt`` and ``task_template`` (``harvey_lab.agent.prompts``).
They are the only text the agent is steered with, and ``HarveyLabAgent`` already
takes both as constructor overrides, so a candidate bundle reaches the real
agent without touching recipe code.

One case = one Harvey LAB task:

* ``input`` — the task id, title, instructions and requested deliverable
  filenames (see ``.rilix-ai/export_dataset.py``).
* ``ground_truth`` — that task's own rubric: ~40-70 binary ``match_criteria``.
* rollout — the task folder is fetched from ``harveyai/harvey-labs`` at the
  pinned commit, then the Stirrup agent runs it under the candidate prompts
  with its single ``code_exec`` tool and submits deliverables through ``finish``.
* score — the recipe's batched rubric judge grades every criterion, and the
  scorer aggregates those verdicts into the two LAB metrics: ``all_pass``
  (1.0 iff every criterion passed) and ``criterion_pass_rate`` (the share that
  passed). The optimizer maximizes ``criterion_pass_rate``, because all-pass is
  far too sparse a signal on a 60-criterion task to steer a search.

Grading runs inside ``run_case`` (as ``harvey-lab evaluate`` does) so the
per-criterion verdicts can ride along as reflection evidence; ``score_case``
stays deterministic and only aggregates them.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from rilixai import (
    Case,
    CaseDataLoader,
    CaseResult,
    CaseScore,
    DatasetRowContext,
    DatasetSchema,
    OptimizationContext,
    OptimizationTargets,
    Spec,
    objective_score,
    optimization_targets_from_prompts,
    spec,
)
from rilixai.sdk import RolloutContext

from harvey_lab.agent.agent import HarveyLabAgent, HarveyLabAgentOutput
from harvey_lab.agent.workspace import task_source_from_dir
from harvey_lab.config import HARVEY_LABS_COMMIT, HarveyLabConfig
from harvey_lab.data.dataset import HarveyLabRecord, load_records
from harvey_lab.data.fetch import ensure_task_dirs
from harvey_lab.evaluation.scoring import (
    ALL_PASS_FIELD,
    CRITERION_PASS_RATE_FIELD,
    build_rubric_judge,
    score_rubric,
)


SPEC_NAME = "harvey-lab"

# The prompt targets, named after the ``HarveyLabAgent`` keyword arguments they
# are threaded into.
SYSTEM_PROMPT_TARGET = "system_prompt"
TASK_TEMPLATE_TARGET = "task_template"

# Scored fields: Harvey LAB's two headline metrics.
FIELD_NAMES = (ALL_PASS_FIELD, CRITERION_PASS_RATE_FIELD)
# all_pass is reported per case but carries no weight in the objective: on a
# ~60-criterion rubric it is 0 for almost every candidate, so optimizing it
# directly gives the search no gradient.
OBJECTIVE_WEIGHTS = {CRITERION_PASS_RATE_FIELD: 1.0}

# Deliverable text is far too large to ship in the trajectory; reflection only
# needs enough of it to see what the agent actually produced.
_PREVIEW_CHARS = 800


# ── Dataset loading ──────────────────────────────────────────────────────────

DATASET_SCHEMA = DatasetSchema(
    json_schema={
        "type": "object",
        "required": ["id", "input", "expected"],
        "properties": {
            "id": {"type": "string", "minLength": 1, "description": "LAB task id (path under the repo's tasks/)."},
            "input": {
                "type": "object",
                "required": ["task_id", "instructions"],
                "properties": {
                    "task_id": {"type": "string", "minLength": 1},
                    "title": {"type": "string"},
                    "work_type": {"type": "string"},
                    "instructions": {"type": "string", "minLength": 1},
                    "deliverables": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Requested output filename -> canonical name; graded by exact filename.",
                    },
                    "documents": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": True,
            },
            "expected": {
                "type": "object",
                "required": ["criteria"],
                "properties": {
                    "criteria": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["id", "match_criteria"],
                            "properties": {
                                "id": {"type": "string", "minLength": 1},
                                "title": {"type": "string"},
                                "match_criteria": {"type": "string", "minLength": 1},
                                "deliverables": {"type": "array", "items": {"type": "string"}},
                            },
                            "additionalProperties": True,
                        },
                    }
                },
                "additionalProperties": True,
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "practice_area": {"type": "string"},
                    "harvey_labs_commit": {"type": "string"},
                    "task_fingerprint": {"type": "string"},
                },
                "additionalProperties": True,
            },
            "group_key": {"type": "string", "description": "Practice area, so scores group the way LAB is drawn."},
        },
        "additionalProperties": True,
    }
)


@dataclass(frozen=True)
class LabTaskRow:
    """One validated JSONL row: a LAB task plus its rubric."""

    task_id: str
    title: str
    work_type: str
    instructions: str
    deliverables: dict[str, str]
    documents: tuple[str, ...]
    criteria: tuple[dict[str, Any], ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def practice_area(self) -> str:
        return str(self.metadata.get("practice_area") or self.task_id.split("/", 1)[0])


class LabTaskDataLoader(CaseDataLoader[LabTaskRow]):
    """Validate ``.rilix-ai/dataset/*.jsonl`` rows and map them to cases.

    Layout (``config_defaults.local_dataset_path`` for dry-runs, an uploaded
    dataset revision for hosted runs)::

        .rilix-ai/dataset/train.jsonl      # required
        .rilix-ai/dataset/val.jsonl        # required
        .rilix-ai/dataset/test.jsonl       # optional
        .rilix-ai/dataset/manifest.json    # provenance: source repo + commit

    Rows are produced by ``.rilix-ai/export_dataset.py`` from the frozen
    ``harvey_lab`` splits; a row without a scoreable criterion is rejected here
    rather than silently scoring 0 (the recipe treats such a task as
    unscoreable and excludes it from its own averages).
    """

    dataset_schema = DATASET_SCHEMA

    def parse_row(self, raw: Mapping[str, Any], context: DatasetRowContext) -> LabTaskRow:
        del context
        payload = _require_mapping(raw.get("input"), "input")
        expected = _require_mapping(raw.get("expected"), "expected")
        metadata = _require_mapping(raw.get("metadata") or {}, "metadata")
        task_id = str(raw.get("id") or payload.get("task_id") or "").strip()
        if not task_id:
            raise ValueError("id must be a non-empty LAB task id")
        instructions = str(payload.get("instructions") or "").strip()
        if not instructions:
            raise ValueError(f"{task_id}: input.instructions must be non-empty")
        criteria = _parse_criteria(expected.get("criteria"), task_id)
        deliverables = _require_mapping(payload.get("deliverables") or {}, "input.deliverables")
        return LabTaskRow(
            task_id=task_id,
            title=str(payload.get("title") or ""),
            work_type=str(payload.get("work_type") or ""),
            instructions=instructions,
            deliverables={str(k): str(v) for k, v in deliverables.items()},
            documents=tuple(str(name) for name in payload.get("documents") or ()),
            criteria=criteria,
            metadata=dict(metadata),
        )

    def iter_cases(self, row: LabTaskRow, context: DatasetRowContext) -> Iterable[Case]:
        del context
        yield Case(
            input={
                "task_id": row.task_id,
                "title": row.title,
                "work_type": row.work_type,
                "instructions": row.instructions,
                "deliverables": dict(row.deliverables),
                "documents": list(row.documents),
            },
            case_id=row.task_id,
            ground_truth={"criteria": [dict(criterion) for criterion in row.criteria]},
            group_key=row.practice_area,
            metadata=dict(row.metadata),
        )


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _parse_criteria(value: Any, task_id: str) -> tuple[dict[str, Any], ...]:
    """Validate the rubric: a non-empty list of binary criteria with standards."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{task_id}: expected.criteria must be a JSON array")
    criteria: list[dict[str, Any]] = []
    for index, entry in enumerate(value):
        entry_map = _require_mapping(entry, f"{task_id}: expected.criteria[{index}]")
        match_criteria = str(entry_map.get("match_criteria") or "").strip()
        if not match_criteria:
            raise ValueError(f"{task_id}: expected.criteria[{index}].match_criteria must be non-empty")
        criteria.append(
            {
                "id": str(entry_map.get("id") or f"C-{index + 1:03d}"),
                "title": str(entry_map.get("title") or ""),
                "match_criteria": match_criteria,
                "deliverables": [str(name) for name in entry_map.get("deliverables") or ()],
            }
        )
    if not criteria:
        raise ValueError(f"{task_id}: expected.criteria is empty, so the task is unscoreable")
    return tuple(criteria)


# ── Rollout ──────────────────────────────────────────────────────────────────


def _seed_targets() -> OptimizationTargets:
    """The prompts the agent runs on today, straight from the recipe."""
    from harvey_lab.agent.prompts import load_harvey_lab_prompts

    system_prompt, task_template = load_harvey_lab_prompts()
    return optimization_targets_from_prompts(
        {SYSTEM_PROMPT_TARGET: system_prompt, TASK_TEMPLATE_TARGET: task_template}
    )


def _config_for_runtime(runtime: RolloutContext | Any) -> HarveyLabConfig:
    """The recipe's production config, with RilixAI's model only when it picks one.

    ``runtime.model`` is a LiteLLM model string, which is exactly what
    ``HarveyLabConfig.task_model`` already is, so model selection needs no
    gateway and no change to the recipe. Without it, the run uses the recipe's
    own default model — prompt-only optimization must not silently re-route the
    agent to another model.
    """
    config = HarveyLabConfig()
    selected_model = getattr(runtime, "model", None)
    if selected_model:
        return replace(config, task_model=str(selected_model))
    return config


def _record_for_case(case: Case, *, cache_dir: Path | None = None) -> tuple[HarveyLabRecord, Path]:
    """Fetch the case's LAB task folder at the pinned commit and load its record.

    The documents are the task, so they cannot live in the JSONL row: the row's
    ``task_id`` is resolved against ``harveyai/harvey-labs`` at
    ``metadata.harvey_labs_commit`` (cached, resumable) exactly like
    ``harvey-lab run``. The rubric that grades the rollout is the row's frozen
    copy, not the freshly read ``task.json``, so a benchmark edit can never move
    the labels underneath a dataset revision — a fingerprint mismatch fails the
    case instead.
    """
    payload = case.input if isinstance(case.input, Mapping) else {}
    task_id = str(payload.get("task_id") or case.case_id)
    commit = str(case.metadata.get("harvey_labs_commit") or HARVEY_LABS_COMMIT)
    tasks_root = ensure_task_dirs([task_id], commit=commit, cache_dir=cache_dir)
    record = load_records(tasks_root, task_ids=[task_id])[0]
    expected_fingerprint = str(case.metadata.get("task_fingerprint") or "")
    if expected_fingerprint and expected_fingerprint != record.task_fingerprint:
        raise RuntimeError(
            f"{task_id}: fetched task tree does not match the dataset row "
            f"(fingerprint {record.task_fingerprint[:12]} != {expected_fingerprint[:12]}); "
            "re-export the dataset from the pinned commit."
        )
    return record, tasks_root


def _placeholder_free_fragment(prompt: str) -> str:
    """The longest chunk of ``prompt`` with no ``{{placeholder}}`` in it."""
    fragments = [chunk.split("}}")[-1] for chunk in prompt.split("{{")]
    return max(fragments, key=len).strip()


def _assert_targets_applied(agent: HarveyLabAgent, prompts: Mapping[str, str]) -> None:
    """Fail loudly if a candidate prompt never reached the agent.

    Cheap insurance against the classic silent failure: optimizing prompts the
    application does not actually use. ``task_template`` is stored verbatim and
    rendered per task; ``system_prompt`` is rendered once at construction
    (``{{max_turns}}`` and the tool names), so it is checked by its longest
    placeholder-free fragment.
    """
    if agent.task_template != prompts[TASK_TEMPLATE_TARGET]:
        raise RuntimeError("the task_template target did not reach HarveyLabAgent.")
    fragment = _placeholder_free_fragment(prompts[SYSTEM_PROMPT_TARGET])
    if fragment and fragment not in agent.system_prompt:
        raise RuntimeError("the system_prompt target did not reach HarveyLabAgent.")


def _task_description(case: Case, record: HarveyLabRecord) -> str:
    payload = case.input if isinstance(case.input, Mapping) else {}
    title = str(payload.get("title") or record.title).strip()
    header = f"{title}\n\n" if title else ""
    return f"{header}{record.instructions}".strip()


def _grade(
    *,
    case: Case,
    output: HarveyLabAgentOutput,
    record: HarveyLabRecord,
    config: HarveyLabConfig,
) -> dict[str, Any]:
    """Grade the submitted deliverables against the row's frozen rubric."""
    criteria = [dict(criterion) for criterion in case.ground_truth.get("criteria", ())]
    judge = build_rubric_judge(
        config.judge_model,
        timeout=config.llm_timeout,
        num_retries=config.judge_num_retries,
    )
    return score_rubric(
        criteria=criteria,
        deliverables=output.deliverables,
        task_description=_task_description(case, record),
        judge=judge,
        batch_size=config.judge_batch_size,
    )


def _trace_evidence(
    *,
    case: Case,
    output: HarveyLabAgentOutput,
    graded: Mapping[str, Any],
    criteria: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compact per-case evidence the optimizer reflects on.

    The useful signal for a prompt rewrite is *which* rubric criteria the work
    product missed and whether the agent even submitted the requested files, so
    those are attributed to the prompt that governs them: submission mechanics
    to the system prompt, task/deliverable framing to the task template.
    """
    titles = {str(criterion.get("id")): str(criterion.get("title") or "") for criterion in criteria}
    failed = [
        f"{verdict.get('id')}: {titles.get(str(verdict.get('id')), '')}".strip()
        for verdict in graded.get("verdicts", ())
        if not verdict.get("passed")
    ]
    submission_notes = [
        f"finished={output.finished} abandoned={output.abandoned} "
        f"max_turns_reached={output.max_turns_reached} turns={output.total_turns}"
    ]
    if output.missing_deliverables:
        submission_notes.append(
            f"never submitted through finish: {', '.join(sorted(output.missing_deliverables))} "
            "(a missing file fails every criterion scoped to it)"
        )
    rubric_notes = [f"{graded.get('passed')}/{graded.get('total_criteria')} rubric criteria passed"]
    if failed:
        rubric_notes.append("failed criteria: " + "; ".join(failed[:25]))
    return {
        "case_id": case.case_id,
        "input_summary": _task_description(case, _EMPTY_RECORD)[:500],
        "per_prompt_feedback": {
            SYSTEM_PROMPT_TARGET: submission_notes,
            TASK_TEMPLATE_TARGET: rubric_notes,
        },
        "application_notes": [
            f"deliverables produced: {', '.join(sorted(output.deliverables)) or '(none)'}",
            f"agent wall time: {output.wall_seconds:.1f}s",
        ],
    }


# Placeholder record used only to render an input summary from the case payload.
_EMPTY_RECORD = HarveyLabRecord(
    task_id="",
    practice_area="",
    title="",
    work_type="",
    instructions="",
    deliverables={},
    criteria=(),
    documents=(),
    raw_task={},
    task_fingerprint="",
)


async def _run_case(*, case: Case, targets: OptimizationTargets, runtime: Any) -> CaseResult:
    """Run one LAB task under the candidate prompts, then grade it."""
    prompts = targets.to_dict()
    config = _config_for_runtime(runtime)
    try:
        record, tasks_root = await asyncio.to_thread(_record_for_case, case)
        agent = HarveyLabAgent(
            config=config,
            task_source=task_source_from_dir(tasks_root),
            system_prompt=prompts[SYSTEM_PROMPT_TARGET],
            task_template=prompts[TASK_TEMPLATE_TARGET],
        )
        _assert_targets_applied(agent, prompts)
        output = await agent.forward(record=record)
        graded = await asyncio.to_thread(_grade, case=case, output=output, record=record, config=config)
    except Exception as exc:  # noqa: BLE001 - a failed rollout scores 0, it must not abort the run
        return CaseResult.failed(f"{type(exc).__name__}: {exc}", retryable=False)

    criteria = [dict(criterion) for criterion in case.ground_truth.get("criteria", ())]
    return CaseResult(
        output={
            ALL_PASS_FIELD: graded[ALL_PASS_FIELD],
            CRITERION_PASS_RATE_FIELD: graded[CRITERION_PASS_RATE_FIELD],
            "passed": graded["passed"],
            "total_criteria": graded["total_criteria"],
            "deliverables_produced": sorted(output.deliverables),
            "deliverables_missing": sorted(output.missing_deliverables),
            "finished": output.finished,
            "abandoned": output.abandoned,
            "max_turns_reached": output.max_turns_reached,
            "total_turns": output.total_turns,
            "final_answer": output.final_answer,
        },
        run_metrics={
            "trace_evidence": _trace_evidence(case=case, output=output, graded=graded, criteria=criteria),
            "timing": {"agent_seconds": round(output.wall_seconds, 3)},
        },
        context={
            # The verdicts the scorer aggregates, plus enough of the work product
            # to make a verdict legible during reflection.
            "verdicts": list(graded.get("verdicts", ())),
            "deliverable_previews": {
                name: text[:_PREVIEW_CHARS] for name, text in sorted(output.deliverables.items())
            },
        },
    )


# ── Scoring ──────────────────────────────────────────────────────────────────


class RubricScorer:
    """Aggregate the judge's per-criterion verdicts into the LAB metrics.

    Deterministic by design: the semantic judgment happened in ``run_case``, so
    this only counts verdicts. A case that never produced verdicts (a failed
    rollout) scores 0 on both fields — a real failure must deflate the metrics,
    never be excluded from them.
    """

    async def score_case(self, *, case: Case, result: CaseResult) -> CaseScore:
        criteria = [dict(criterion) for criterion in case.ground_truth.get("criteria", ())]
        total = len(criteria)
        verdicts = result.context.get("verdicts") or ()
        passed_ids = {str(verdict.get("id")) for verdict in verdicts if verdict.get("passed")}
        passed = sum(1 for criterion in criteria if str(criterion.get("id")) in passed_ids)
        rate = (passed / total) if total else 0.0
        scores = {
            ALL_PASS_FIELD: 1.0 if total and passed == total else 0.0,
            CRITERION_PASS_RATE_FIELD: rate,
        }
        return CaseScore(
            field_scores=scores,
            objective=objective_score(scores, field_weights=OBJECTIVE_WEIGHTS),
            key="harvey-lab-rubric",
        )


# ── Spec ─────────────────────────────────────────────────────────────────────


@spec(
    name=SPEC_NAME,
    version="v1",
    description="Harvey LAB legal agent: optimize the LAB-AA system prompt + task template",
    metadata={"task_type": SPEC_NAME},
    dataset_schema=DATASET_SCHEMA,
)
def build_spec(ctx: OptimizationContext) -> Spec:
    """Assemble the Harvey LAB prompt-optimization spec."""
    del ctx
    return Spec(
        name=SPEC_NAME,
        seed_targets=_seed_targets(),
        data_loader=LabTaskDataLoader(),
        run_case=_run_case,
        scorer=RubricScorer(),
    )
