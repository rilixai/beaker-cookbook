"""Beaker prompt-optimization spec for the Harvey LAB legal agent."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from beaker import (
    Case,
    CaseDataLoader,
    CaseResult,
    CaseScore,
    DatasetRowContext,
    DatasetSchema,
    OptimizationContext,
    OptimizationTargets,
    Spec,
    inference_target,
    spec,
)


METRIC_FIELDS = ("all_pass", "criterion_pass_rate")
OBJECTIVE_FIELD = "criterion_pass_rate"
PROMPT_FIELDS = ("system_prompt", "task_template")

HARVEY_LAB_DATASET_SCHEMA = DatasetSchema(
    json_schema={
        "type": "object",
        "required": ["id", "input", "expected"],
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "input": {
                "type": "object",
                "required": ["task_id"],
                "properties": {"task_id": {"type": "string", "minLength": 1}},
                "additionalProperties": False,
            },
            "expected": {
                "type": "object",
                "required": list(METRIC_FIELDS),
                "properties": {name: {"type": "number", "const": 1.0} for name in METRIC_FIELDS},
                "additionalProperties": False,
            },
            "metadata": {"type": "object"},
            "group_key": {"type": "string", "minLength": 1},
        },
        "additionalProperties": False,
    }
)


@dataclass(frozen=True)
class _HarveyLabRow:
    id: str
    task_id: str
    expected: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    group_key: str = "default"


class _HarveyLabDataLoader(CaseDataLoader[_HarveyLabRow]):
    """Map frozen public LAB task IDs to optimization cases.

    A task ID resolves at rollout time to the pinned ``harveyai/harvey-labs``
    task directory. Its ``task.json`` supplies the instructions, requested
    deliverables, and labeled rubric criteria; its ``documents/`` tree supplies
    the case inputs. The expected metric values describe a fully passing result.
    """

    dataset_schema = HARVEY_LAB_DATASET_SCHEMA

    def parse_row(self, raw: Mapping[str, Any], context: DatasetRowContext) -> _HarveyLabRow:
        del context
        missing = [name for name in ("id", "input", "expected") if name not in raw]
        if missing:
            raise ValueError(f"missing required field(s): {', '.join(missing)}")

        row_id = str(raw["id"]).strip()
        input_payload = raw["input"]
        expected = raw["expected"]
        metadata = raw.get("metadata") or {}
        if not row_id:
            raise ValueError("id must be non-empty")
        if not isinstance(input_payload, Mapping):
            raise TypeError("input must be a JSON object")
        if not isinstance(expected, Mapping):
            raise TypeError("expected must be a JSON object")
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a JSON object")

        task_id = str(input_payload.get("task_id") or "").strip()
        if not task_id:
            raise ValueError("input.task_id must be a non-empty string")
        normalized_expected: dict[str, float] = {}
        for field_name in METRIC_FIELDS:
            value = expected.get(field_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"expected.{field_name} must be numeric")
            if float(value) != 1.0:
                raise ValueError(f"expected.{field_name} must be 1.0")
            normalized_expected[field_name] = float(value)

        return _HarveyLabRow(
            id=row_id,
            task_id=task_id,
            expected=normalized_expected,
            metadata=dict(metadata),
            group_key=str(raw.get("group_key") or task_id.partition("/")[0] or "default"),
        )

    def iter_cases(self, row: _HarveyLabRow, context: DatasetRowContext) -> Iterable[Case]:
        del context
        yield Case(
            input={"task_id": row.task_id},
            case_id=row.id,
            ground_truth=row.expected,
            group_key=row.group_key,
            metadata=row.metadata,
        )


def _load_record(task_id: str) -> tuple[Any, Any]:
    """Resolve one frozen task and return its record plus workspace factory."""
    from harvey_lab.agent.workspace import task_source_from_dir
    from harvey_lab.data.dataset import load_records
    from harvey_lab.data.fetch import ensure_task_dirs

    tasks_root = ensure_task_dirs([task_id])
    record = load_records(tasks_root, task_ids=[task_id])[0]
    return record, task_source_from_dir(tasks_root)


def _gateway_model_factory(runtime: Any) -> Any:
    """Adapt Beaker's OpenAI-compatible inference target to Stirrup."""
    target = inference_target(runtime)

    def _factory(
        _model: str,
        temperature: float,
        max_tokens: int,
        context_window_tokens: int,
        timeout: float,
        reasoning_effort: str,
    ) -> Any:
        from stirrup.clients.litellm_client import LiteLLMClient, ReasoningEffort

        effort = reasoning_effort if reasoning_effort not in ("", "none") else None
        return LiteLLMClient(
            model=target.model,
            max_tokens=max_tokens,
            context_window_tokens=context_window_tokens,
            api_key=target.api_key,
            reasoning_effort=cast("ReasoningEffort | None", effort),
            kwargs={
                "api_base": target.base_url,
                "custom_llm_provider": "openai",
                "temperature": temperature,
                "timeout": timeout,
            },
        )

    return _factory


async def _run_case(*, case: Case, targets: OptimizationTargets, runtime: Any) -> CaseResult:
    """Run one real legal-agent rollout with Beaker's candidate prompts."""
    from beaker.tracing.integrations import litellm as litellm_tracing

    from harvey_lab.agent.agent import HarveyLabAgent
    from harvey_lab.config import HarveyLabConfig

    task_id = str(case.input["task_id"])
    prompts = targets.to_dict()
    missing_prompts = [name for name in PROMPT_FIELDS if name not in prompts]
    if missing_prompts:
        return CaseResult.failed(f"candidate is missing prompt(s): {', '.join(missing_prompts)}")

    try:
        record, task_source = _load_record(task_id)
        agent = HarveyLabAgent(
            config=HarveyLabConfig(),
            task_source=task_source,
            model_factory=_gateway_model_factory(runtime) if runtime.model else None,
            system_prompt=prompts["system_prompt"],
            task_template=prompts["task_template"],
        )
        with runtime.trace.stage(
            "harvey_lab.agent_rollout",
            inputs={"task_id": task_id, "practice_area": record.practice_area},
        ) as stage:
            # Stirrup calls LiteLLM internally. The scoped adapter records every
            # turn without changing Stirrup's client or global production state.
            async with litellm_tracing.registered(runtime.trace) as litellm_trace:
                output = await agent.forward(record=record)
                await litellm_trace.flush()
            stage.output(
                {
                    "finished": output.finished,
                    "abandoned": output.abandoned,
                    "deliverables_produced": sorted(output.deliverables),
                    "deliverables_missing": sorted(output.missing_deliverables),
                    "total_turns": output.total_turns,
                }
            )
    except Exception as error:
        return CaseResult.failed(f"{type(error).__name__}: {error}", retryable=True)

    return CaseResult(
        output={"deliverables": output.deliverables},
        context={
            "task_id": task_id,
            "final_answer": output.final_answer,
            "submitted_paths": output.submitted_paths,
            "finished": output.finished,
            "abandoned": output.abandoned,
            "max_turns_reached": output.max_turns_reached,
            "total_turns": output.total_turns,
            "wall_seconds": output.wall_seconds,
            "deliverables_missing": output.missing_deliverables,
        },
    )


class _HarveyLabScorer:
    """Grade produced deliverables with LAB's labeled binary rubric."""

    async def score_case(self, *, case: Case, result: CaseResult) -> CaseScore:
        from harvey_lab.agent.agent import HarveyLabAgentOutput
        from harvey_lab.config import HarveyLabConfig
        from harvey_lab.evaluation.run_eval import evaluate_output
        from harvey_lab.evaluation.scoring import build_rubric_judge

        task_id = str(case.input["task_id"])
        record, _task_source = _load_record(task_id)
        payload = result.output if isinstance(result.output, Mapping) else {}
        deliverables = payload.get("deliverables") or {}
        if not isinstance(deliverables, Mapping):
            deliverables = {}
        context = result.context if isinstance(result.context, Mapping) else {}
        agent_output = HarveyLabAgentOutput(
            final_answer=str(context.get("final_answer") or ""),
            deliverables={str(name): str(value) for name, value in deliverables.items()},
            missing_deliverables=[str(name) for name in context.get("deliverables_missing") or ()],
            submitted_paths=[str(path) for path in context.get("submitted_paths") or ()],
            finished=bool(context.get("finished")),
            abandoned=bool(context.get("abandoned")),
            max_turns_reached=bool(context.get("max_turns_reached")),
            total_turns=int(context.get("total_turns") or 0),
            wall_seconds=float(context.get("wall_seconds") or 0.0),
        )
        config = HarveyLabConfig()
        scored = await evaluate_output(
            record=record,
            output=agent_output,
            judge=build_rubric_judge(
                model=config.judge_model,
                timeout=config.judge_llm_timeout,
                num_retries=config.judge_num_retries,
            ),
            batch_size=config.judge_batch_size,
        )
        scores = {
            "all_pass": max(0.0, min(1.0, float(scored.get("all_pass", 0.0)))),
            "criterion_pass_rate": max(0.0, min(1.0, float(scored.get("criterion_pass_rate", 0.0)))),
        }
        return CaseScore(field_scores=scores, objective=scores[OBJECTIVE_FIELD], key=OBJECTIVE_FIELD)


def _seed_targets() -> OptimizationTargets:
    from harvey_lab.agent.prompts import SYSTEM_PROMPT_SEED, TASK_TEMPLATE_SEED

    return OptimizationTargets(
        prompts={
            "system_prompt": SYSTEM_PROMPT_SEED,
            "task_template": TASK_TEMPLATE_SEED,
        }
    )


@spec(dataset_schema=HARVEY_LAB_DATASET_SCHEMA, repository=None)
def build_spec(ctx: OptimizationContext) -> Spec:
    """Optimize both prompts against criterion pass rate on public LAB tasks."""
    del ctx
    return Spec(
        data_loader=_HarveyLabDataLoader(),
        run_case=_run_case,
        scorer=_HarveyLabScorer(),
        seed_targets=_seed_targets(),
    )
