"""Harvey LAB's real Stirrup agent and rubric, exposed through Beaker.

Setup alone fetches the pinned corpus. Only documents cross into candidate
execution; task.json and its grading rubric stay on the trusted side.
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import tempfile
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from beaker import (
    Case,
    CaseFile,
    CaseResult,
    CaseScore,
    Check,
    Integration,
    JsonValue,
    RepositoryRunSetup,
    RepositoryRunSetupResult,
    RetryableCaseError,
    RolloutRuntime,
    SetupRuntime,
    inference_target,
    repository,
    scoring_inference_target,
)
from beaker.tracing.integrations.litellm import registered
from pydantic import BaseModel, Field, field_validator, model_validator

from harvey_lab.config import HARVEY_LABS_COMMIT, HarveyLabConfig
from harvey_lab.data.dataset import HarveyLabRecord, load_records
from harvey_lab.data.fetch import ensure_task_dirs
from harvey_lab.evaluation.scoring import (
    ALL_PASS_FIELD,
    CRITERION_PASS_RATE_FIELD,
    JudgeCallError,
    _extract_verdicts_payload,
    build_rubric_judge,
    score_rubric,
)


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or path.as_posix() != value or ".." in path.parts or "\\" in value:
        raise ValueError("expected a normalized relative POSIX path")
    return value


class TaskInput(BaseModel):
    task_id: str
    title: str
    instructions: str = Field(min_length=1)
    work_type: str = ""
    deliverables: dict[str, str]
    commit: str = HARVEY_LABS_COMMIT

    _task_path = field_validator("task_id")(relative_path)

    @field_validator("commit")
    @classmethod
    def pinned_commit(cls, value: str) -> str:
        if value != HARVEY_LABS_COMMIT:
            raise ValueError("dataset must use the recipe's pinned HARVEY_LABS_COMMIT")
        return value

    @field_validator("deliverables")
    @classmethod
    def safe_deliverables(cls, value: dict[str, str]) -> dict[str, str]:
        for name in value:
            relative_path(name)
        return value


class Criterion(BaseModel):
    id: str = Field(min_length=1)
    title: str
    match_criteria: str = Field(min_length=1)
    deliverables: list[str]


class Expected(BaseModel):
    criteria: list[Criterion] = Field(min_length=1)

    @field_validator("criteria")
    @classmethod
    def unique_criteria(cls, value: list[Criterion]) -> list[Criterion]:
        if len({criterion.id for criterion in value}) != len(value):
            raise ValueError("rubric criterion IDs must be unique")
        if any(not criterion.match_criteria.strip() for criterion in value):
            raise ValueError("rubric criteria must contain grading instructions")
        return value


class TaskRow(BaseModel):
    id: str = Field(min_length=1)
    input: TaskInput
    expected: Expected
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def matching_id(self) -> TaskRow:
        if self.id != self.input.task_id:
            raise ValueError("row id must match input.task_id")
        return self


class TaskSetup(RepositoryRunSetup[TaskRow]):
    row_model = TaskRow

    @asynccontextmanager
    async def prepare_run(self, *, runtime: SetupRuntime) -> AsyncIterator[RepositoryRunSetupResult]:
        del runtime
        with tempfile.TemporaryDirectory(prefix="harvey-beaker-corpus-") as cache:
            self.cache_dir = Path(cache)
            yield RepositoryRunSetupResult()

    async def load_cases(self, row: TaskRow, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        # Optional local source avoids network in smoke; never forwarded to a candidate.
        local_root = runtime.config.get("tasks_root")
        if local_root is not None:
            if not isinstance(local_root, str):
                raise ValueError("tasks_root must be a path string")
            tasks_root = Path(local_root)
        else:
            tasks_root = await asyncio.to_thread(
                ensure_task_dirs, [row.id], commit=row.input.commit, cache_dir=self.cache_dir
            )
        record = (await asyncio.to_thread(load_records, tasks_root, task_ids=[row.id]))[0]
        # Reject stale labels or prompts instead of evaluating a different task.
        source_input, source_expected = row_payload(record)
        if row.input != source_input or row.expected != source_expected:
            raise ValueError(f"dataset row differs from the pinned task: {row.id}")
        files_dir = runtime.case_files_dir(row.id)
        documents_dir = tasks_root / row.id / "documents"
        files = []
        for name in record.documents:
            relative_path(name)
            source = documents_dir / name
            if not source.resolve().is_relative_to(documents_dir.resolve()):
                raise ValueError(f"document escapes task folder: {name}")
            target = files_dir / "documents" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(shutil.copyfile, source, target)
            files.append(CaseFile(name=f"documents/{name}", path=target))
        yield Case(
            id=row.id,
            input=row.input.model_dump(mode="json"),
            expected=row.expected.model_dump(mode="json"),
            files=tuple(files),
            metadata=row.metadata,
        )


def row_payload(record: HarveyLabRecord) -> tuple[TaskInput, Expected]:
    return (
        TaskInput(
            task_id=record.task_id,
            title=record.title,
            work_type=record.work_type,
            instructions=record.instructions,
            deliverables=dict(record.deliverables),
        ),
        Expected(criteria=[Criterion(**vars(criterion)) for criterion in record.criteria]),
    )


def selected_model_factory(runtime: RolloutRuntime[Any]) -> Any:
    """Use Stirrup's native client and the selected model's Beaker gateway."""
    from stirrup.clients.litellm_client import LiteLLMClient

    target = inference_target(runtime)

    def factory(
        model: str,
        temperature: float,
        max_tokens: int,
        context_window_tokens: int,
        timeout: float,
        reasoning_effort: str,
    ) -> Any:
        del model, reasoning_effort
        return LiteLLMClient(
            model=f"openai/{target.model}",
            max_tokens=max_tokens,
            context_window_tokens=context_window_tokens,
            api_key=target.api_key,
            reasoning_effort=None,
            kwargs={"api_base": target.base_url, "temperature": temperature, "timeout": timeout},
        )

    return factory


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime[Any]) -> CaseResult:
    # Import here: each repository candidate supplies its own agent implementation.
    import litellm

    from harvey_lab.agent.agent import HarveyLabAgent
    from harvey_lab.agent.workspace import TaskWorkspace

    task = TaskInput.model_validate(case_input)
    documents = runtime.case_files_dir / "documents"
    record = HarveyLabRecord(
        task_id=task.task_id,
        practice_area=task.task_id.split("/", 1)[0],
        title=task.title,
        work_type=task.work_type,
        instructions=task.instructions,
        deliverables=task.deliverables,
        criteria=(),
        documents=tuple(path.relative_to(documents).as_posix() for path in documents.rglob("*") if path.is_file()),
        raw_task=task.model_dump(mode="json"),
        task_fingerprint=hashlib.sha256(task.model_dump_json().encode()).hexdigest(),
    )
    with tempfile.TemporaryDirectory(prefix="harvey-beaker-case-") as directory:
        workspace = TaskWorkspace(directory)
        if documents.is_dir():
            await asyncio.to_thread(shutil.copytree, documents, workspace.documents_dir, dirs_exist_ok=True)
        agent = HarveyLabAgent(
            config=HarveyLabConfig(),
            task_source=lambda _: workspace,
            model_factory=selected_model_factory(runtime) if runtime.model else None,
        )
        with runtime.trace.stage("harvey_lab.run_case", inputs={"task_id": task.task_id}) as stage:
            # Stirrup calls LiteLLM for every turn and compaction. The registered
            # adapter covers only this candidate workflow, never the rubric judge.
            async with registered(runtime.trace) as tracing:
                try:
                    output = await agent.forward(record=record)
                except (
                    litellm.Timeout,
                    litellm.RateLimitError,
                    litellm.APIConnectionError,
                    litellm.InternalServerError,
                ) as exc:
                    raise RetryableCaseError(str(exc)) from exc
                finally:
                    await tracing.flush()
            stage.output({"finished": output.finished, "abandoned": output.abandoned, "turns": output.total_turns})
        # Only the submitted documents are predictions. Binary copies and the
        # conversation can be large; extracted text is the benchmark's scoring input.
        return CaseResult(output=dict(output.deliverables), output_kind="text")


def grade(task: TaskInput, expected: Expected, deliverables: dict[str, str]) -> CaseScore:
    import litellm

    config = HarveyLabConfig()
    target = scoring_inference_target()  # Missing hosted scorer configuration must fail.
    reasons: dict[str, str] = {}

    def call_llm(*, model: str, messages: list[dict[str, str]]) -> str:
        routing = {"api_base": target.base_url, "api_key": target.api_key} if target else {}
        response = litellm.completion(
            model=f"openai/{target.model}" if target else model,
            messages=messages,
            temperature=0.0,
            timeout=config.judge_llm_timeout,
            num_retries=config.judge_num_retries,
            **routing,
        )
        text = str(response.choices[0].message.content or "")
        payload = _extract_verdicts_payload(text)
        if payload and isinstance(payload.get("verdicts"), list):
            for entry in payload["verdicts"]:
                if isinstance(entry, Mapping) and entry.get("reasoning"):
                    reasons[str(entry.get("id"))] = str(entry["reasoning"])
        return text

    judge = build_rubric_judge(config.judge_model, llm=call_llm)
    failures: dict[tuple[str, ...], Exception] = {}

    def checked_judge(description: str, criteria: Sequence[Mapping[str, Any]], output: str) -> dict[str, bool]:
        key = tuple(str(criterion["id"]) for criterion in criteria)
        try:
            verdicts = judge(description, criteria, output)
        except Exception as exc:
            failures[key] = exc
            raise
        failures.pop(key, None)
        return verdicts

    scored = score_rubric(
        criteria=[criterion.model_dump(mode="json") for criterion in expected.criteria],
        deliverables=deliverables,
        task_description=f"{task.title}\n\n{task.instructions}".strip(),
        judge=checked_judge,
        batch_size=config.judge_batch_size,
    )
    # The standalone harness converts exhausted judge failures to FAIL. Beaker
    # must surface infrastructure failures instead of optimizing against false zeros.
    if failures:
        raise JudgeCallError("Rubric judge failed after retries") from next(iter(failures.values()))
    verdicts = {str(item["id"]): item["passed"] for item in scored["verdicts"]}
    checks = []
    for criterion in expected.criteria:
        missing = bool(criterion.deliverables) and not any(name in deliverables for name in criterion.deliverables)
        checks.append(
            Check(
                name=criterion.title or criterion.id,
                description=criterion.match_criteria,
                verdict="pass" if verdicts[criterion.id] else "fail",
                group=", ".join(criterion.deliverables) or "All deliverables",
                message="No required deliverable was submitted." if missing else reasons.get(criterion.id),
            )
        )
    return CaseScore(
        objective=scored[CRITERION_PASS_RATE_FIELD],
        field_scores={name: scored[name] for name in (CRITERION_PASS_RATE_FIELD, ALL_PASS_FIELD)},
        checks=tuple(checks),
    )


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    del case_files_dir
    task = TaskInput.model_validate(case.input)
    expected = Expected.model_validate(case.expected)
    if not isinstance(result.output, dict) or not all(isinstance(value, str) for value in result.output.values()):
        raise ValueError("Harvey output must map submitted filenames to extracted text")
    deliverables = {name: value for name, value in result.output.items() if isinstance(value, str)}
    return await asyncio.to_thread(grade, task, expected, deliverables)


integration = Integration(
    targets=repository(paths=("src/harvey_lab/agent",)),
    run_setup=TaskSetup,
    run_case=run_case,
    score_case=score_case,
)
