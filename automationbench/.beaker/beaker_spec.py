"""Beaker repository optimization spec for AutomationBench skills."""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from beaker import (
    STANDARD_JSONL_CASE_SCHEMA,
    Case,
    CaseDataLoader,
    CaseResult,
    CaseScore,
    DatasetRowContext,
    InferenceTarget,
    OptimizationContext,
    Spec,
    inference_target,
    spec,
)
from verifiers.clients import Client

from automationbench_skills.data.tasks import load_samples
from automationbench_skills.runner import DEFAULT_MODEL, ModelSpec, get_client, run_one_async


FIELD_NAMES = ("partial_credit", "task_completed_correctly")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = PROJECT_ROOT / "skills"
BEAKER_API_KEY_VAR = "BEAKER_AUTOMATIONBENCH_API_KEY"


@dataclass(frozen=True)
class _AutomationBenchRow:
    id: str
    input: dict[str, Any]
    expected: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    group_key: str = "default"


class _AutomationBenchDataLoader(CaseDataLoader[_AutomationBenchRow]):
    dataset_schema = STANDARD_JSONL_CASE_SCHEMA

    def parse_row(self, raw: Mapping[str, Any], context: DatasetRowContext) -> _AutomationBenchRow:
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
        task_name = input_payload.get("task_name")
        if not isinstance(task_name, str) or not task_name.strip():
            raise ValueError("input.task_name must be a non-empty string")
        if not isinstance(expected, Mapping):
            raise TypeError("expected must be a JSON object")
        for name in FIELD_NAMES:
            if name not in expected or float(expected[name]) != 1.0:
                raise ValueError(f"expected.{name} must be 1.0")
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a JSON object")
        return _AutomationBenchRow(
            id=row_id,
            input={"task_name": task_name.strip()},
            expected=dict(expected),
            metadata=dict(metadata),
            group_key=str(raw.get("group_key") or "default"),
        )

    def iter_cases(self, row: _AutomationBenchRow, context: DatasetRowContext) -> Iterable[Case]:
        del context
        yield Case(
            input=row.input,
            case_id=row.id,
            ground_truth=row.expected,
            group_key=row.group_key,
            metadata=row.metadata,
        )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    return str(value)


class _TracingClient(Client[Any, Any, Any, Any]):
    def __init__(self, delegate: Client, runtime: Any, provider: str) -> None:
        self._delegate = delegate
        self._runtime = runtime
        self._provider = provider
        super().__init__(delegate.client)

    def setup_client(self, config: Any) -> Any:
        return self._delegate.setup_client(config)

    async def to_native_tool(self, tool: Any) -> Any:
        return await self._delegate.to_native_tool(tool)

    async def to_native_prompt(self, messages: Any) -> tuple[Any, dict[str, Any]]:
        return await self._delegate.to_native_prompt(messages)

    async def get_native_response(
        self,
        prompt: Any,
        model: str,
        sampling_args: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self._delegate.get_native_response(prompt, model, sampling_args, tools, **kwargs)

    async def raise_from_native_response(self, response: Any) -> None:
        await self._delegate.raise_from_native_response(response)

    async def from_native_response(self, response: Any) -> Any:
        return await self._delegate.from_native_response(response)

    async def close(self) -> None:
        await self._delegate.close()

    async def get_response(
        self,
        prompt: Any,
        model: str,
        sampling_args: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        with self._runtime.trace.model_call(
            operation="chat",
            provider=self._provider,
            model=model,
            input_messages=_jsonable(prompt),
        ) as call:
            response = await self._delegate.get_response(prompt, model, sampling_args, tools, **kwargs)
            call.output(_jsonable(response))
            return response


def _application_default_gateway_target() -> InferenceTarget | None:
    base_url = os.environ.get("BEAKER_INFERENCE_BASE_URL")
    api_key = os.environ.get("BEAKER_INFERENCE_API_KEY")
    if not base_url and not api_key:
        return None
    if not base_url or not api_key:
        raise RuntimeError("Beaker application inference credentials are incomplete")
    return InferenceTarget(base_url=base_url.rstrip("/"), api_key=api_key, model=f"openai:{DEFAULT_MODEL}")


def _model_for_runtime(runtime: Any) -> tuple[ModelSpec, str, str | None]:
    target = inference_target(runtime) if runtime.model else _application_default_gateway_target()
    if target is None:
        return ModelSpec(name=DEFAULT_MODEL), "openai", None
    provider = target.model.split(":", 1)[0] if ":" in target.model else "openai"
    reasoning_effort = runtime.model_capabilities.reasoning_effort if runtime.model_capabilities else None
    model = ModelSpec(
        name=target.model,
        base_url=target.base_url,
        api_key_var=BEAKER_API_KEY_VAR,
        api="chat_completions",
        reasoning_effort=reasoning_effort,
    )
    return model, provider, target.api_key


def _sample(task_name: str) -> Any:
    sample = next((item for item in load_samples() if item.task_name == task_name), None)
    if sample is None:
        raise ValueError(f"Unknown AutomationBench task: {task_name}")
    return sample


async def _run_case(*, case: Case, targets: None, runtime: Any) -> CaseResult:
    del targets
    if not isinstance(case.input, Mapping):
        return CaseResult.failed("case input must be a JSON object")
    task_name = case.input.get("task_name")
    if not isinstance(task_name, str) or not task_name:
        return CaseResult.failed("case input.task_name must be a non-empty string")

    try:
        sample = _sample(task_name)
        model, provider, api_key = _model_for_runtime(runtime)
        previous_api_key = os.environ.get(BEAKER_API_KEY_VAR)
        if api_key is not None:
            os.environ[BEAKER_API_KEY_VAR] = api_key
        try:
            client = get_client(model)
            traced_client = _TracingClient(client, runtime, provider)
            with runtime.trace.stage(
                "automationbench.run_task",
                inputs={"task_name": task_name, "domain": sample.domain},
            ) as stage:
                result = await run_one_async(sample, model=model, skills_dir=SKILLS_DIR, client=traced_client)
                output = {
                    "partial_credit": result.partial_credit,
                    "task_completed_correctly": result.task_completed_correctly,
                }
                stage.output(output)
        finally:
            if api_key is not None:
                if previous_api_key is None:
                    os.environ.pop(BEAKER_API_KEY_VAR, None)
                else:
                    os.environ[BEAKER_API_KEY_VAR] = previous_api_key
    except Exception as exc:
        return CaseResult.failed(str(exc), retryable=True)

    if result.error is not None:
        return CaseResult.failed(str(result.error), context=result.to_json(), retryable=True)
    return CaseResult(output=output, context=result.to_json())


def _metric(output: Mapping[str, Any], name: str) -> float:
    try:
        value = float(output.get(name, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, value)) if math.isfinite(value) else 0.0


class _AutomationBenchScorer:
    async def score_case(self, *, case: Case, result: CaseResult) -> CaseScore:
        del case
        output = result.output if isinstance(result.output, Mapping) else {}
        scores = {name: _metric(output, name) for name in FIELD_NAMES}
        return CaseScore(field_scores=scores, objective=scores["partial_credit"], key="automationbench")


@spec(
    dataset_schema=STANDARD_JSONL_CASE_SCHEMA,
    repository=("skills",),
)
def build_spec(ctx: OptimizationContext) -> Spec:
    del ctx
    return Spec(
        data_loader=_AutomationBenchDataLoader(),
        run_case=_run_case,
        scorer=_AutomationBenchScorer(),
    )
