"""Review all public ALE tasks; execute through the unmodified upstream runner."""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from beaker import (
    Case,
    CaseResult,
    CaseScore,
    Check,
    Integration,
    JsonValue,
    RolloutRuntime,
    RunSetup,
    SetupRuntime,
    repository,
)
from benchmark_data import COMMIT, Row, TaskInput
from runner import run_task


class Setup(RunSetup[Row]):
    row_model = Row

    async def load_cases(self, row: Row, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        del runtime
        yield Case(id=row.id, input=row.input.model_dump(), expected=row.expected, metadata=row.metadata)


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime[Any]) -> CaseResult:
    task = TaskInput.model_validate(case_input)
    if runtime.model:
        raise ValueError("ALE uses the model in agent.yaml; model comparison is not configured")
    start = asyncio.get_running_loop().time()
    with runtime.trace.stage("ale.task", inputs={"task_id": task.task_id}) as stage:
        result = await run_task(task.task_id, commit=COMMIT)
        stage.output(result)
    return CaseResult(
        output=result,
        output_kind="record",
        context={
            "beaker_rollout_duration_ms": (asyncio.get_running_loop().time() - start) * 1000,
        },
    )


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    del case_files_dir
    output = result.output
    if not isinstance(output, dict) or output.get("task_id") != case.id or output.get("eval_status") != "success":
        raise ValueError("ALE did not complete evaluation for this task")
    score = output.get("score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        raise ValueError("ALE must return a finite upstream score in [0, 1]")
    return CaseScore(
        objective=float(score),
        field_scores={"task_score": float(score)},
        checks=(
            Check(
                name="Upstream ALE evaluation",
                verdict="pass" if score == 1 else "fail",
                expected=1,
                predicted=score,
                message="Unmodified upstream task evaluator",
            ),
        ),
    )


integration = Integration(
    targets=repository(("src/agents_last_exam/agent.yaml",)), run_setup=Setup, run_case=run_case, score_case=score_case
)
