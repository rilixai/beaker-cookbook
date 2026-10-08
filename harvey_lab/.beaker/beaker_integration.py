"""Harvey LAB's real Stirrup agent and rubric, exposed through Beaker.

Setup alone fetches the pinned corpus. Only documents cross into candidate
execution; task.json and its grading rubric stay on the trusted side.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import tempfile
import zipfile
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


BEAKER_JUDGE_BATCH_SIZE = 4
SELECTED_CONTEXT_WINDOW_ENV = "HARVEY_BEAKER_CONTEXT_WINDOW_TOKENS"
DEFAULT_SELECTED_CONTEXT_WINDOW = 128_000


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
        selected_context_window()
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
        files = await asyncio.to_thread(pack_documents, documents_dir, record.documents, files_dir)
        yield Case(
            id=row.id,
            input=row.input.model_dump(mode="json"),
            expected=row.expected.model_dump(mode="json"),
            files=tuple(files),
            metadata=row.metadata,
        )


def pack_documents(documents_dir: Path, names: Sequence[str], destination: Path) -> tuple[CaseFile, ...]:
    """Carry large data rooms within Beaker's 128-file / 64-MiB-per-file limits.

    Bound each archive by 32 MiB of source bytes, leaving room for ZIP headers.
    Fixed timestamps make the archives reproducible across setup attempts.
    """
    groups: list[list[tuple[str, Path]]] = []
    size = 0
    for name in sorted(names):
        relative_path(name)
        source = documents_dir / name
        if not source.resolve().is_relative_to(documents_dir.resolve()):
            raise ValueError(f"document escapes task folder: {name}")
        source_size = source.stat().st_size
        if source_size > 60 * 1024 * 1024:
            raise ValueError(f"document is too large for a case archive: {name}")
        if not groups or size + source_size > 32 * 1024 * 1024:
            groups.append([])
            size = 0
        groups[-1].append((name, source))
        size += source_size
    files = []
    for index, group in enumerate(groups):
        path = destination / f"documents-{index:03d}.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, source in group:
                info = zipfile.ZipInfo(name)
                info.compress_type = zipfile.ZIP_DEFLATED
                with source.open("rb") as stream, archive.open(info, "w") as output:
                    shutil.copyfileobj(stream, output)
        files.append(CaseFile(name=path.name, path=path))
    return tuple(files)


def unpack_documents(case_files_dir: Path, destination: Path) -> tuple[str, ...]:
    """Restore the original document paths without exposing task.json or labels."""
    names: list[str] = []
    seen: set[str] = set()
    for path in sorted(case_files_dir.glob("documents-*.zip")):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                name = relative_path(member.filename)
                if name in seen or member.is_dir():
                    raise ValueError(f"invalid or duplicate archived document: {name}")
                seen.add(name)
                target = destination / name
                if not target.resolve().is_relative_to(destination.resolve()):
                    raise ValueError(f"archived document escapes workspace: {name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                names.append(name)
    return tuple(names)


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


def selected_context_window() -> int:
    """Validate the optional model window before setup downloads or case execution."""
    raw_window = os.environ.get(SELECTED_CONTEXT_WINDOW_ENV, str(DEFAULT_SELECTED_CONTEXT_WINDOW))
    try:
        selected_window = int(raw_window)
    except ValueError as exc:
        raise ValueError(f"{SELECTED_CONTEXT_WINDOW_ENV} must be a positive integer") from exc
    if selected_window <= 0:
        raise ValueError(f"{SELECTED_CONTEXT_WINDOW_ENV} must be a positive integer")
    return selected_window


def selected_model_factory(runtime: RolloutRuntime[Any]) -> Any:
    """Use Stirrup's native client and the selected model's Beaker gateway."""
    from stirrup.clients.litellm_client import LiteLLMClient

    selected_window = selected_context_window()
    target = inference_target(runtime)

    def factory(
        model: str,
        temperature: float,
        max_tokens: int,
        context_window_tokens: int,
        timeout: float,
        reasoning_effort: str,
    ) -> Any:
        del model, temperature, max_tokens, context_window_tokens, reasoning_effort
        # The SDK target supplies no model limits. Use the configured model window.
        # Stirrup requires a numeric budget locally; omit it on the wire so
        # the gateway chooses the selected model's output cap.
        return LiteLLMClient(
            model=f"openai/{target.model}",
            max_tokens=min(32_768, selected_window),
            context_window_tokens=selected_window,
            api_key=target.api_key,
            reasoning_effort=None,
            kwargs={"api_base": target.base_url, "timeout": timeout, "additional_drop_params": ["max_tokens"]},
        )

    return factory


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime[Any]) -> CaseResult:
    # Import here: each repository candidate supplies its own agent implementation.
    import litellm
    from tenacity import RetryError

    from harvey_lab.agent.agent import HarveyLabAgent
    from harvey_lab.agent.workspace import TaskWorkspace

    transient_errors = (
        litellm.Timeout,
        litellm.RateLimitError,
        litellm.APIConnectionError,
        litellm.InternalServerError,
    )
    task = TaskInput.model_validate(case_input)
    with tempfile.TemporaryDirectory(prefix="harvey-beaker-case-") as directory:
        workspace = TaskWorkspace(directory)
        documents = await asyncio.to_thread(unpack_documents, runtime.case_files_dir, workspace.documents_dir)
        record = HarveyLabRecord(
            task_id=task.task_id,
            practice_area=task.task_id.split("/", 1)[0],
            title=task.title,
            work_type=task.work_type,
            instructions=task.instructions,
            deliverables=task.deliverables,
            criteria=(),
            documents=documents,
            raw_task=task.model_dump(mode="json"),
            task_fingerprint=hashlib.sha256(task.model_dump_json().encode()).hexdigest(),
        )
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
                except transient_errors as exc:
                    raise RetryableCaseError(str(exc)) from exc
                except RetryError as exc:
                    cause = exc.last_attempt.exception()
                    if isinstance(cause, transient_errors):
                        raise RetryableCaseError(str(cause)) from exc
                    raise
                finally:
                    await tracing.flush()
            stage.output({"finished": output.finished, "abandoned": output.abandoned, "turns": output.total_turns})
        # Only the submitted documents are predictions. Binary copies and the
        # conversation can be large; extracted text is the benchmark's scoring input.
        return CaseResult(output=dict(output.deliverables), output_kind="text")


def complete_judge_reasons(text: str, ids: Sequence[str]) -> dict[str, str]:
    """Reject missing, duplicate, unknown, or invalid verdicts before aggregation."""
    payload = _extract_verdicts_payload(text)
    entries = payload.get("verdicts") if payload else None
    if not isinstance(entries, list):
        raise JudgeCallError("Rubric judge did not return a verdict list")
    wanted = set(ids)
    seen: set[str] = set()
    reasons: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise JudgeCallError("Rubric judge returned a malformed verdict")
        cid = entry.get("id")
        if not isinstance(cid, str) or cid not in wanted or cid in seen:
            raise JudgeCallError("Rubric judge returned an unknown or duplicate criterion ID")
        verdict = entry.get("verdict")
        if not isinstance(verdict, str) or verdict.strip().lower() not in {"pass", "fail"}:
            raise JudgeCallError(f"Rubric judge returned an invalid verdict for {cid}")
        seen.add(cid)
        if entry.get("reasoning"):
            reasons[cid] = str(entry["reasoning"])
    if seen != wanted:
        raise JudgeCallError(f"Rubric judge returned {len(seen)}/{len(wanted)} verdicts; retry required")
    return reasons


def grade(task: TaskInput, expected: Expected, deliverables: dict[str, str]) -> CaseScore:
    import litellm

    config = HarveyLabConfig(judge_batch_size=BEAKER_JUDGE_BATCH_SIZE)
    target = scoring_inference_target()  # Missing hosted scorer configuration must fail.
    reasons: dict[str, str] = {}

    failures: dict[tuple[str, ...], Exception] = {}

    def checked_judge(description: str, criteria: Sequence[Mapping[str, Any]], output: str) -> dict[str, bool]:
        key = tuple(str(criterion["id"]) for criterion in criteria)

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
            # The standalone parser fills omitted criteria with FAIL. Validate
            # the raw reply first so those defaults never become Beaker scores.
            reasons.update(complete_judge_reasons(text, key))
            return text

        judge = build_rubric_judge(config.judge_model, llm=call_llm)
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
