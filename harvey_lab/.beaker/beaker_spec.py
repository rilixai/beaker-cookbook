"""Beaker repository optimization spec for the Harvey LAB legal agent.

The candidate is the live agent package (``src/harvey_lab/agent``): prompts,
tool wiring, and workspace staging. Evaluation policy stays here: load a frozen
Harvey LAB task, run ``HarveyLabAgent.forward``, and score the batched rubric
judge's ``criterion_pass_rate``.

Contract: the dataset row's ``expected`` holds the task's rubric
(``{"criteria": [...]}``), ``run_case`` returns the produced deliverable text in
``output``, and the scorer emits one ``Check`` per criterion so the optimizer
sees which requirements failed.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from beaker import (
    STANDARD_JSONL_CASE_SCHEMA,
    Case,
    CaseDataLoader,
    CaseResult,
    CaseScore,
    Check,
    DatasetRowContext,
    OptimizationContext,
    Spec,
    inference_target,
    objective_score,
    scoring_inference_target,
    spec,
)
from beaker.tracing import current_trace
from beaker.tracing.integrations.litellm import registered

from harvey_lab.agent.agent import HarveyLabAgent
from harvey_lab.agent.workspace import task_source_from_dir
from harvey_lab.config import HarveyLabConfig
from harvey_lab.data.dataset import HarveyLabRecord, RubricCriterion, load_records
from harvey_lab.data.fetch import ensure_task_dirs
from harvey_lab.evaluation.scoring import (
    ALL_PASS_FIELD,
    CRITERION_PASS_RATE_FIELD,
    DEFAULT_JUDGE_BATCH_SIZE,
    DEFAULT_JUDGE_MODEL,
    build_rubric_judge,
    score_rubric,
)


FIELD_WEIGHTS = {CRITERION_PASS_RATE_FIELD: 1.0, ALL_PASS_FIELD: 0.0}
LLM_SCORER_MODEL = "openrouter:deepseek/deepseek-v4-flash"
LOCAL_JUDGE_MODEL = DEFAULT_JUDGE_MODEL


@dataclass(frozen=True)
class _HarveyLabRow:
    id: str
    input: dict[str, Any]
    expected: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    group_key: str = "default"


class _HarveyLabDataLoader(CaseDataLoader[_HarveyLabRow]):
    """Validate Harvey LAB JSONL rows produced by ``upload_splits.py``."""

    dataset_schema = STANDARD_JSONL_CASE_SCHEMA

    def parse_row(self, raw: Mapping[str, Any], context: DatasetRowContext) -> _HarveyLabRow:
        del context
        missing = [name for name in ("id", "input", "expected") if name not in raw]
        if missing:
            raise ValueError(f"missing required field(s): {', '.join(missing)}")
        row_id = str(raw["id"]).strip()
        if not row_id:
            raise ValueError("id must be non-empty")
        input_payload = raw["input"]
        expected = raw["expected"]
        metadata = raw.get("metadata") or {}
        if not isinstance(input_payload, Mapping):
            raise TypeError("input must be a JSON object")
        if not isinstance(expected, Mapping):
            raise TypeError("expected must be a JSON object")
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a JSON object")
        if not str(input_payload.get("task_id") or row_id).strip():
            raise ValueError("input.task_id is required")
        if not str(input_payload.get("instructions") or "").strip():
            raise ValueError("input.instructions is required")
        criteria = expected.get("criteria")
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("expected.criteria must be a non-empty list")
        if not all(isinstance(c, Mapping) and str(c.get("match_criteria") or "").strip() for c in criteria):
            raise TypeError("expected.criteria must be objects with match_criteria")
        return _HarveyLabRow(
            id=row_id,
            input=dict(input_payload),
            expected=dict(expected),
            metadata=dict(metadata),
            group_key=str(raw.get("group_key") or input_payload.get("practice_area") or "default"),
        )

    def iter_cases(self, row: _HarveyLabRow, context: DatasetRowContext) -> Iterable[Case]:
        del context
        yield Case(
            input=row.input,
            case_id=row.id,
            ground_truth=row.expected,
            group_key=row.group_key,
            metadata=row.metadata,
        )


def _model_factory_for_runtime(runtime: Any):
    """Point Stirrup's LiteLLM client at the Beaker gateway when a model is selected."""
    if not getattr(runtime, "model", None):
        return None

    target = inference_target(runtime)

    def _factory(
        model: str,
        temperature: float,
        max_tokens: int,
        context_window_tokens: int,
        timeout: float,
        reasoning_effort: str,
    ) -> Any:
        del model
        from stirrup.clients.litellm_client import LiteLLMClient, ReasoningEffort

        effort = reasoning_effort if reasoning_effort not in ("", "none") else None
        gateway_model = target.model if str(target.model).startswith("openai/") else f"openai/{target.model}"
        kwargs: dict[str, Any] = {
            "temperature": temperature,
            "timeout": timeout,
            "api_base": target.base_url,
        }
        if effort is not None:
            kwargs["allowed_openai_params"] = ["reasoning_effort"]
        return LiteLLMClient(
            model=gateway_model,
            max_tokens=max_tokens,
            context_window_tokens=context_window_tokens,
            api_key=target.api_key,
            reasoning_effort=cast("ReasoningEffort | None", effort),
            kwargs=kwargs,
        )

    return _factory


def _tasks_root_for_case(task_id: str) -> Path:
    """Ensure the task tree is cached and return its ``tasks/`` root."""
    override = os.environ.get("HARVEY_LAB_TASKS_ROOT", "").strip()
    if override:
        root = Path(override)
        if not (root / task_id / "task.json").is_file():
            raise FileNotFoundError(f"HARVEY_LAB_TASKS_ROOT missing task {task_id}")
        return root
    return ensure_task_dirs([task_id])


def _record_from_case(case: Case, tasks_root: Path) -> HarveyLabRecord:
    """Rebuild a ``HarveyLabRecord`` from the JSONL case + on-disk task tree."""
    payload = case.input if isinstance(case.input, Mapping) else {}
    task_id = str(payload.get("task_id") or case.case_id)
    loaded = load_records(tasks_root, task_ids=[task_id])
    if not loaded:
        raise FileNotFoundError(f"Could not load task {task_id} from {tasks_root}")
    record = loaded[0]

    raw_criteria = case.ground_truth.get("criteria") if isinstance(case.ground_truth, Mapping) else None
    criteria = record.criteria
    if isinstance(raw_criteria, list) and raw_criteria:
        criteria = tuple(
            RubricCriterion(
                id=str(c.get("id") or f"C-{idx + 1:03d}"),
                title=str(c.get("title") or ""),
                match_criteria=str(c.get("match_criteria") or ""),
                deliverables=tuple(str(d) for d in (c.get("deliverables") or ()) if d),
            )
            for idx, c in enumerate(raw_criteria)
            if isinstance(c, Mapping) and str(c.get("match_criteria") or "").strip()
        )
    deliverables = payload.get("deliverables")
    return HarveyLabRecord(
        task_id=record.task_id,
        practice_area=str(payload.get("practice_area") or record.practice_area),
        title=str(payload.get("title") or record.title),
        work_type=str(payload.get("work_type") or record.work_type),
        instructions=str(payload.get("instructions") or record.instructions),
        deliverables=dict(deliverables) if isinstance(deliverables, Mapping) else dict(record.deliverables),
        criteria=criteria,
        documents=record.documents,
        raw_task=record.raw_task,
        task_fingerprint=record.task_fingerprint,
    )


async def _run_case(*, case: Case, targets: None, runtime: Any) -> CaseResult:
    """Run one Harvey LAB task against the candidate repository."""
    del targets
    payload = case.input if isinstance(case.input, Mapping) else {}
    task_id = str(payload.get("task_id") or case.case_id)
    with runtime.trace.stage(
        "harvey_lab.forward",
        inputs={"case_id": case.case_id, "task_id": task_id, "title": payload.get("title")},
    ) as stage:
        try:
            tasks_root = _tasks_root_for_case(task_id)
            record = _record_from_case(case, tasks_root)
            agent = HarveyLabAgent(
                config=HarveyLabConfig(),
                task_source=task_source_from_dir(tasks_root),
                model_factory=_model_factory_for_runtime(runtime),
            )
            async with registered(current_trace()) as litellm_trace:
                output = await agent.forward(record=record)
                await litellm_trace.flush()
        except Exception as exc:
            stage.output({"error": f"{type(exc).__name__}: {exc}"})
            return CaseResult.failed(f"{type(exc).__name__}: {exc}", retryable=True)

        stage.output(
            {
                "finished": output.finished,
                "abandoned": output.abandoned,
                "max_turns_reached": output.max_turns_reached,
                "total_turns": output.total_turns,
                "deliverables": sorted(output.deliverables),
                "missing_deliverables": list(output.missing_deliverables),
            }
        )
        return CaseResult(
            output={"deliverables": dict(output.deliverables)},
            output_kind="text",
            context={
                "final_answer": output.final_answer,
                "missing_deliverables": list(output.missing_deliverables),
                "finished": output.finished,
                "abandoned": output.abandoned,
                "max_turns_reached": output.max_turns_reached,
                "total_turns": output.total_turns,
            },
        )


def _task_description(case: Case) -> str:
    payload = case.input if isinstance(case.input, Mapping) else {}
    title = str(payload.get("title") or "").strip()
    instructions = str(payload.get("instructions") or "").strip()
    header = f"{title}\n\n" if title else ""
    return f"{header}{instructions}".strip()


def _criteria_payload(case: Case) -> list[dict[str, Any]]:
    raw = case.ground_truth.get("criteria") if isinstance(case.ground_truth, Mapping) else None
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        match = str(entry.get("match_criteria") or "").strip()
        if not match:
            continue
        out.append(
            {
                "id": str(entry.get("id") or ""),
                "title": str(entry.get("title") or ""),
                "match_criteria": match,
                "deliverables": [str(d) for d in (entry.get("deliverables") or ()) if d],
            }
        )
    return out


def _judge_for_scoring() -> Any:
    """Hosted gateway via ``scoring_inference_target``; local LiteLLM otherwise."""
    target = scoring_inference_target()
    if target is None:
        return build_rubric_judge(model=LOCAL_JUDGE_MODEL)

    def _llm(*, model: str, messages: Sequence[Mapping[str, str]]) -> str:
        import litellm

        gateway_model = model if str(model).startswith("openai/") else f"openai/{model}"
        response = litellm.completion(
            model=gateway_model,
            messages=list(messages),
            temperature=0.0,
            api_base=target.base_url,
            api_key=target.api_key,
        )
        return str(response.choices[0].message.content or "")

    return build_rubric_judge(model=target.model, llm=_llm)


def _check_for_criterion(
    criterion: Mapping[str, Any],
    *,
    passed: bool,
    missing: Sequence[str],
    abandoned: bool,
    error: str | None,
) -> Check:
    title = str(criterion.get("title") or criterion.get("id") or "criterion").strip()
    deliverables = [str(d) for d in (criterion.get("deliverables") or ()) if d]
    if abandoned and not passed:
        message = "agent abandoned the task before producing a complete submission"
    elif error and not passed:
        message = error
    elif not passed and deliverables and all(name in missing for name in deliverables):
        message = "none of this criterion's deliverables were produced"
    elif passed:
        message = None
    else:
        message = "did not satisfy the criterion"
    return Check(
        name=title,
        description=str(criterion.get("match_criteria") or ""),
        verdict="pass" if passed else "fail",
        message=message,
        group=deliverables[0] if deliverables else None,
    )


class _RubricJudgeScorer:
    """Score produced deliverables against the case rubric with the batched LAB judge."""

    async def score_case(self, *, case: Case, result: CaseResult) -> CaseScore:
        output = result.output if isinstance(result.output, Mapping) else {}
        context = result.context if isinstance(result.context, Mapping) else {}
        deliverables = output.get("deliverables") if isinstance(output.get("deliverables"), Mapping) else {}
        missing = [str(name) for name in (context.get("missing_deliverables") or ())]
        abandoned = bool(context.get("abandoned"))
        error = None if context.get("error") is None else str(context["error"])
        criteria = _criteria_payload(case)
        if not criteria:
            scores = {CRITERION_PASS_RATE_FIELD: 0.0, ALL_PASS_FIELD: 0.0}
            return CaseScore(
                field_scores=scores,
                objective=objective_score(scores, field_weights=FIELD_WEIGHTS),
                key="default",
                checks=(),
            )

        judge = _judge_for_scoring()
        scored = await asyncio.to_thread(
            score_rubric,
            criteria=criteria,
            deliverables={str(k): str(v) for k, v in deliverables.items()},
            task_description=_task_description(case),
            judge=judge,
            batch_size=DEFAULT_JUDGE_BATCH_SIZE,
        )
        passed_by_id = {str(v.get("id")): bool(v.get("passed")) for v in scored.get("verdicts") or []}
        scores = {
            CRITERION_PASS_RATE_FIELD: float(scored[CRITERION_PASS_RATE_FIELD]),
            ALL_PASS_FIELD: float(scored[ALL_PASS_FIELD]),
        }
        return CaseScore(
            field_scores=scores,
            objective=objective_score(scores, field_weights=FIELD_WEIGHTS),
            key="default",
            checks=tuple(
                _check_for_criterion(
                    criterion,
                    passed=passed_by_id.get(str(criterion.get("id")), False),
                    missing=missing,
                    abandoned=abandoned,
                    error=error,
                )
                for criterion in criteria
            ),
        )


@spec(
    dataset_schema=STANDARD_JSONL_CASE_SCHEMA,
    repository=("src/harvey_lab/agent",),
)
def build_spec(ctx: OptimizationContext) -> Spec:
    del ctx
    return Spec(
        data_loader=_HarveyLabDataLoader(),
        run_case=_run_case,
        scorer=_RubricJudgeScorer(),
        llm_scorer_model=LLM_SCORER_MODEL,
    )
