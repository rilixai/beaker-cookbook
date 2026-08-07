"""Beaker prompt-optimization spec for the Harvey LAB legal agent.

Optimizes the LAB-AA ``system_prompt`` + ``task_template`` pair. Cases are real
Harvey LAB tasks (rubric criteria are the grading standard). Scoring uses the
repo's batched LLM judge; the headline field is ``criterion_pass_rate``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from beaker import (
    Case,
    CaseDataLoader,
    CaseResult,
    CaseScore,
    DatasetRowContext,
    OptimizationContext,
    OptimizationTargets,
    STANDARD_JSONL_CASE_SCHEMA,
    Spec,
    inference_target,
    objective_score,
    optimization_targets_from_prompts,
    scoring_inference_target,
    spec,
)
from harvey_lab.agent.agent import HarveyLabAgent, HarveyLabAgentOutput
from harvey_lab.agent.prompts import SYSTEM_PROMPT_SEED, TASK_TEMPLATE_SEED
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


# Scored prediction fields (LAB metrics from the batched rubric judge).
FIELD_NAMES = [CRITERION_PASS_RATE_FIELD, ALL_PASS_FIELD]

# Hosted LLM-judge model in Beaker canonical form. Local dry-runs keep the
# repo's established LiteLLM string (DEFAULT_JUDGE_MODEL).
LLM_SCORER_MODEL = "openrouter:deepseek/deepseek-v4-flash"
LOCAL_JUDGE_MODEL = DEFAULT_JUDGE_MODEL

# Optional local dry-run turn cap; unset → HarveyLabConfig.max_turns (200).
_MAX_TURNS_ENV = "BEAKER_HARVEY_MAX_TURNS"


# ── Dataset loading ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _HarveyLabRow:
    id: str
    input: dict[str, Any]
    expected: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    group_key: str = "default"


class _HarveyLabDataLoader(CaseDataLoader[_HarveyLabRow]):
    """Validate Harvey LAB JSONL rows produced by ``prepare_dataset.py``."""

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
        if "task_id" not in input_payload:
            raise ValueError("input.task_id is required")
        criteria = expected.get("criteria")
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("expected.criteria must be a non-empty list")
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


# ── Spec wiring ──────────────────────────────────────────────────────────────


def _seed_targets() -> OptimizationTargets:
    """AA LAB-AA prompts currently used by ``HarveyLabAgent``."""
    return optimization_targets_from_prompts(
        {
            "system_prompt": SYSTEM_PROMPT_SEED,
            "task_template": TASK_TEMPLATE_SEED,
        }
    )


def _config_for_runtime() -> HarveyLabConfig:
    base = HarveyLabConfig()
    raw = os.environ.get(_MAX_TURNS_ENV, "").strip()
    if not raw:
        return base
    return replace(base, max_turns=max(1, int(raw)))


def _model_factory_for_runtime(runtime: Any):
    """Use Beaker gateway only when a model is selected; else production LiteLLM."""
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
        del model  # selected model comes from the Beaker rollout target
        from stirrup.clients.litellm_client import LiteLLMClient, ReasoningEffort
        from typing import cast

        effort = reasoning_effort if reasoning_effort not in ("", "none") else None
        # Beaker gateway is OpenAI Chat Completions; route via litellm openai/*.
        gateway_model = target.model if target.model.startswith("openai/") else f"openai/{target.model}"
        kwargs: dict[str, Any] = {
            "temperature": temperature,
            "timeout": timeout,
            "api_base": target.base_url,
            "api_key": target.api_key,
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
    task_id = str(case.input.get("task_id") or case.case_id)
    loaded = load_records(tasks_root, task_ids=[task_id])
    if not loaded:
        raise FileNotFoundError(f"Could not load task {task_id} from {tasks_root}")
    record = loaded[0]

    # Prefer rubric criteria from the dataset row (upload contract) when present.
    raw_criteria = case.ground_truth.get("criteria") if isinstance(case.ground_truth, Mapping) else None
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
        return HarveyLabRecord(
            task_id=record.task_id,
            practice_area=record.practice_area,
            title=str(case.input.get("title") or record.title),
            work_type=str(case.input.get("work_type") or record.work_type),
            instructions=str(case.input.get("instructions") or record.instructions),
            deliverables=dict(case.input.get("deliverables") or record.deliverables),
            criteria=criteria,
            documents=record.documents,
            raw_task=record.raw_task,
            task_fingerprint=record.task_fingerprint,
        )
    return record


def _output_payload(output: HarveyLabAgentOutput) -> dict[str, Any]:
    return {
        "final_answer": output.final_answer,
        "deliverables": dict(output.deliverables),
        "missing_deliverables": list(output.missing_deliverables),
        "finished": output.finished,
        "abandoned": output.abandoned,
        "max_turns_reached": output.max_turns_reached,
        "total_turns": output.total_turns,
        CRITERION_PASS_RATE_FIELD: None,
        ALL_PASS_FIELD: None,
    }


async def _run_case(*, case: Case, targets: OptimizationTargets, runtime: Any) -> CaseResult:
    """Run one Harvey LAB task with the candidate prompt bundle."""
    prompts = targets.to_dict()
    system_prompt = prompts["system_prompt"]
    task_template = prompts["task_template"]
    task_id = str(case.input.get("task_id") or case.case_id)

    with runtime.trace.stage(
        "harvey_lab.run_case",
        inputs={"case_id": case.case_id, "task_id": task_id},
    ) as stage:
        try:
            tasks_root = _tasks_root_for_case(task_id)
            record = _record_from_case(case, tasks_root)
            config = _config_for_runtime()
            agent = HarveyLabAgent(
                config=config,
                task_source=task_source_from_dir(tasks_root),
                model_factory=_model_factory_for_runtime(runtime),
                system_prompt=system_prompt,
                task_template=task_template,
            )
            stage.attribute("beaker.system_prompt_chars", len(agent.system_prompt))
            stage.attribute("beaker.task_template_chars", len(agent.task_template))
            output = await agent.forward(record=record)
            payload = _output_payload(output)
            stage.output(
                {
                    "finished": output.finished,
                    "abandoned": output.abandoned,
                    "deliverables": sorted(output.deliverables),
                    "total_turns": output.total_turns,
                }
            )
            return CaseResult(output=payload)
        except Exception as exc:
            return CaseResult.failed(str(exc), retryable=True)


def _task_description(case: Case) -> str:
    title = str(case.input.get("title") or "").strip()
    instructions = str(case.input.get("instructions") or "").strip()
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
        out.append(
            {
                "id": str(entry.get("id") or ""),
                "title": str(entry.get("title") or ""),
                "match_criteria": str(entry.get("match_criteria") or ""),
                "deliverables": [str(d) for d in (entry.get("deliverables") or ()) if d],
            }
        )
    return out


def _judge_for_scoring() -> tuple[Any, str]:
    """Hosted gateway via ``scoring_inference_target``; local LiteLLM otherwise."""
    target = scoring_inference_target()
    if target is None:
        return build_rubric_judge(model=LOCAL_JUDGE_MODEL), LOCAL_JUDGE_MODEL

    # Hosted OpenAI-compatible gateway — call through litellm with api_base.
    def _llm(*, model: str, messages: Sequence[Mapping[str, str]]) -> str:
        import litellm

        gateway_model = model if model.startswith("openai/") else f"openai/{model}"
        response = litellm.completion(
            model=gateway_model,
            messages=list(messages),
            temperature=0.0,
            api_base=target.base_url,
            api_key=target.api_key,
        )
        return str(response.choices[0].message.content or "")

    return build_rubric_judge(model=target.model, llm=_llm), target.model


class _RubricJudgeScorer:
    """Score deliverables against the case rubric with the batched LAB judge."""

    async def score_case(self, *, case: Case, result: CaseResult) -> CaseScore:
        import asyncio

        output = result.output if isinstance(result.output, Mapping) else {}
        deliverables = output.get("deliverables") if isinstance(output.get("deliverables"), Mapping) else {}
        criteria = _criteria_payload(case)
        if not criteria:
            scores = {CRITERION_PASS_RATE_FIELD: 0.0, ALL_PASS_FIELD: 0.0}
            return CaseScore(
                field_scores=scores,
                objective=objective_score(
                    scores,
                    field_weights={CRITERION_PASS_RATE_FIELD: 1.0, ALL_PASS_FIELD: 0.0},
                ),
                key="default",
            )

        judge, _model = _judge_for_scoring()
        scored = await asyncio.to_thread(
            score_rubric,
            criteria=criteria,
            deliverables={str(k): str(v) for k, v in deliverables.items()},
            task_description=_task_description(case),
            judge=judge,
            batch_size=DEFAULT_JUDGE_BATCH_SIZE,
        )
        scores = {
            CRITERION_PASS_RATE_FIELD: float(scored[CRITERION_PASS_RATE_FIELD]),
            ALL_PASS_FIELD: float(scored[ALL_PASS_FIELD]),
        }
        return CaseScore(
            field_scores=scores,
            objective=objective_score(
                scores,
                field_weights={CRITERION_PASS_RATE_FIELD: 1.0, ALL_PASS_FIELD: 0.0},
            ),
            key="default",
        )


@spec(
    dataset_schema=STANDARD_JSONL_CASE_SCHEMA,
)
def build_spec(ctx: OptimizationContext) -> Spec:
    """Assemble the Harvey LAB prompt-optimization spec."""
    del ctx
    return Spec(
        seed_targets=_seed_targets(),
        data_loader=_HarveyLabDataLoader(),
        run_case=_run_case,
        scorer=_RubricJudgeScorer(),
        llm_scorer_model=LLM_SCORER_MODEL,
    )
