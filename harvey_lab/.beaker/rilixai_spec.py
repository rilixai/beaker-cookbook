"""RilixAI optimization spec for the Harvey LAB legal agent.

Dataset rows reference real tasks in Harvey's public ``harvey-labs`` repository
at the commit pinned by this package.  The loader materializes each task and its
documents, the candidate prompt bundle drives the real Stirrup agent, and the
existing LAB rubric judge supplies the optimization score.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
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
    RolloutContext,
    Spec,
    inference_target,
    objective_score,
    optimization_targets_from_prompts,
    spec,
)

from harvey_lab.agent.agent import HarveyLabAgent, HarveyLabAgentOutput
from harvey_lab.agent.prompts import SYSTEM_PROMPT_SEED, TASK_TEMPLATE_SEED
from harvey_lab.agent.workspace import task_source_from_dir
from harvey_lab.config import HARVEY_LABS_COMMIT, HarveyLabConfig
from harvey_lab.data.dataset import HarveyLabRecord, load_records
from harvey_lab.data.fetch import ensure_task_dirs
from harvey_lab.evaluation.scoring import (
    ALL_PASS_FIELD,
    CRITERION_PASS_RATE_FIELD,
    BatchJudge,
    build_rubric_judge,
    score_rubric,
)


SYSTEM_PROMPT_TARGET = "system_prompt"
TASK_TEMPLATE_TARGET = "task_template"

HARVEY_LAB_DATASET_SCHEMA = DatasetSchema(
    json_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["task_id", "source_commit"],
        "properties": {
            "task_id": {
                "type": "string",
                "minLength": 1,
                "description": "Path of a real task under harveyai/harvey-labs/tasks.",
            },
            "source_commit": {
                "type": "string",
                "const": HARVEY_LABS_COMMIT,
                "description": "Pinned public benchmark revision used for labels and documents.",
            },
        },
    },
)


@dataclass(frozen=True)
class HarveyLabDatasetRow:
    """One immutable reference to a labeled public Harvey LAB task."""

    task_id: str


def _criterion_payload(record: HarveyLabRecord) -> list[dict[str, Any]]:
    return [
        {
            "id": criterion.id,
            "title": criterion.title,
            "match_criteria": criterion.match_criteria,
            "deliverables": list(criterion.deliverables),
        }
        for criterion in record.criteria
    ]


def _load_record(task_id: str) -> tuple[Path, HarveyLabRecord]:
    tasks_root = ensure_task_dirs([task_id], commit=HARVEY_LABS_COMMIT)
    record = load_records(tasks_root, task_ids=[task_id])[0]
    return tasks_root, record


class HarveyLabDataLoader(CaseDataLoader[HarveyLabDatasetRow]):
    """Resolve pinned task references into real LAB inputs and rubric labels."""

    dataset_schema = HARVEY_LAB_DATASET_SCHEMA

    def parse_row(self, raw: Mapping[str, Any], context: DatasetRowContext) -> HarveyLabDatasetRow:
        del context
        task_id = str(raw.get("task_id") or "").strip()
        parts = task_id.split("/")
        if not task_id or task_id.startswith("/") or "\\" in task_id or any(part in ("", ".", "..") for part in parts):
            raise ValueError("task_id must be a non-empty relative Harvey LAB task path")
        source_commit = str(raw.get("source_commit") or "").strip()
        if source_commit != HARVEY_LABS_COMMIT:
            raise ValueError(
                f"source_commit must be the pinned Harvey LAB revision {HARVEY_LABS_COMMIT}, got {source_commit!r}"
            )
        return HarveyLabDatasetRow(task_id=task_id)

    def iter_cases(self, row: HarveyLabDatasetRow, context: DatasetRowContext) -> Iterable[Case]:
        del context
        _tasks_root, record = _load_record(row.task_id)
        criteria = _criterion_payload(record)
        if not criteria:
            raise ValueError(f"Harvey LAB task {row.task_id!r} has no scoreable rubric criteria")
        yield Case(
            input={"task_id": row.task_id},
            case_id=row.task_id,
            ground_truth={"criteria": criteria, "source_commit": HARVEY_LABS_COMMIT},
            group_key=record.practice_area,
            metadata={
                "title": record.title,
                "instructions": record.instructions,
                "deliverables": dict(record.deliverables),
                "task_fingerprint": record.task_fingerprint,
            },
        )


def _seed_targets() -> OptimizationTargets:
    """Return the two prompts that currently drive every Harvey LAB rollout."""
    return optimization_targets_from_prompts(
        {
            SYSTEM_PROMPT_TARGET: SYSTEM_PROMPT_SEED,
            TASK_TEMPLATE_TARGET: TASK_TEMPLATE_SEED,
        }
    )


def _gateway_model_factory(runtime: RolloutContext) -> Any:
    """Adapt RilixAI's OpenAI-compatible rollout target to Stirrup/LiteLLM."""
    target = inference_target(runtime)

    def _factory(
        _production_model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        reasoning_effort: str,
    ) -> Any:
        from stirrup.clients.litellm_client import LiteLLMClient

        effort = reasoning_effort if reasoning_effort not in ("", "none") else None
        return LiteLLMClient(
            # Force LiteLLM's OpenAI-compatible transport while preserving the
            # canonical provider:model identifier expected by the gateway.
            model=f"openai/{target.model}",
            max_tokens=max_tokens,
            api_key=target.api_key,
            reasoning_effort=effort,
            kwargs={
                "api_base": target.base_url,
                "temperature": temperature,
                "timeout": timeout,
                "allowed_openai_params": ["reasoning_effort"],
            },
        )

    return _factory


def _prediction(output: HarveyLabAgentOutput) -> dict[str, Any]:
    return {
        "final_answer": output.final_answer,
        "deliverables": dict(output.deliverables),
        "missing_deliverables": list(output.missing_deliverables),
        "finished": output.finished,
        "abandoned": output.abandoned,
        "max_turns_reached": output.max_turns_reached,
        "total_turns": output.total_turns,
    }


async def _run_case(*, case: Case, targets: OptimizationTargets, runtime: RolloutContext) -> CaseResult:
    """Run the real Harvey LAB agent with the candidate prompt bundle."""
    prompts = targets.to_dict()
    missing_targets = [name for name in (SYSTEM_PROMPT_TARGET, TASK_TEMPLATE_TARGET) if name not in prompts]
    if missing_targets:
        return CaseResult.failed(f"missing prompt target(s): {', '.join(missing_targets)}")
    task_id = str(case.input.get("task_id") if isinstance(case.input, Mapping) else "").strip()
    if not task_id:
        return CaseResult.failed("case.input.task_id is required")

    with runtime.trace.stage(
        "harvey_lab.run_task",
        inputs={"task_id": task_id, "prompt_targets": sorted(prompts)},
    ) as stage:
        try:
            tasks_root, record = await asyncio.to_thread(_load_record, task_id)
            model_factory = _gateway_model_factory(runtime) if runtime.model else None
            agent = HarveyLabAgent(
                config=HarveyLabConfig(),
                task_source=task_source_from_dir(tasks_root),
                model_factory=model_factory,
                system_prompt=prompts[SYSTEM_PROMPT_TARGET],
                task_template=prompts[TASK_TEMPLATE_TARGET],
            )
            output = await agent.forward(record=record)
        except Exception as exc:  # A failed fetch/harness/model call produced no gradable work product.
            message = f"{type(exc).__name__}: {exc}"
            stage.output({"status": "failed", "error": message})
            return CaseResult.failed(message, retryable=True)

        prediction = _prediction(output)
        stage.attribute("harvey_lab.total_turns", output.total_turns)
        stage.attribute("harvey_lab.wall_seconds", output.wall_seconds)
        stage.output(
            {
                "status": "finished" if output.finished else "incomplete",
                "deliverables_produced": sorted(output.deliverables),
                "deliverables_missing": sorted(output.missing_deliverables),
            }
        )
        return CaseResult(output=prediction)


class HarveyLabRubricScorer:
    """Score submitted work products with the repository's LAB rubric judge."""

    def __init__(self, judge: BatchJudge | None = None) -> None:
        self._judge = judge or build_rubric_judge()

    async def score_case(self, *, case: Case, result: CaseResult) -> CaseScore:
        output = result.output if isinstance(result.output, Mapping) else {}
        deliverables = output.get("deliverables") or {}
        if not isinstance(deliverables, Mapping):
            raise TypeError("result.output.deliverables must be an object")
        criteria = case.ground_truth.get("criteria") or []
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("case.ground_truth.criteria must be a non-empty list")
        title = str(case.metadata.get("title") or "").strip()
        instructions = str(case.metadata.get("instructions") or "").strip()
        task_description = f"{title}\n\n{instructions}".strip()
        scored = await asyncio.to_thread(
            score_rubric,
            criteria=criteria,
            deliverables={str(name): str(text) for name, text in deliverables.items()},
            task_description=task_description,
            judge=self._judge,
        )
        field_scores = {
            ALL_PASS_FIELD: float(scored[ALL_PASS_FIELD]),
            CRITERION_PASS_RATE_FIELD: float(scored[CRITERION_PASS_RATE_FIELD]),
        }
        return CaseScore(
            field_scores=field_scores,
            objective=objective_score(field_scores, field_weights={CRITERION_PASS_RATE_FIELD: 1.0}),
            key="harvey_lab_rubric",
        )


@spec(
    name="harvey-labs",
    version="v1",
    description="Optimize the Harvey LAB legal agent's system and task prompts against real public-task rubrics.",
    metadata={"task_type": "legal-work-product", "source_commit": HARVEY_LABS_COMMIT},
    dataset_schema=HARVEY_LAB_DATASET_SCHEMA,
)
def build_spec(ctx: OptimizationContext) -> Spec:
    """Build the prompt-optimization contract used locally and by hosted runs."""
    del ctx
    return Spec(
        name="harvey-labs",
        seed_targets=_seed_targets(),
        data_loader=HarveyLabDataLoader(),
        run_case=_run_case,
        scorer=HarveyLabRubricScorer(),
    )
