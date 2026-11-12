"""Optimize a tau2-bench customer-service agent; keep the user simulator and scoring fixed.

Editable: ``prompts/``, the agent loop (``agent.py``) and its chat-call layer
(``llm.py``). Fixed: task data, the pinned user simulator, the orchestrator and
tau2's evaluator, which runs here in ``score_case`` on the recorded trajectory.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import openai
from beaker import (
    Case,
    CaseResult,
    CaseScore,
    Integration,
    JsonValue,
    RetryableCaseError,
    RolloutRuntime,
    RunSetup,
    SetupRuntime,
    inference_target,
    repository,
)

from tau2_bench import benchmark, scoring
from tau2_bench.agent import Agent, load_instructions
from tau2_bench.data import Row, TaskRef
from tau2_bench.llm import ChatModel


# Provider outages are not the candidate's fault: retry the case instead of scoring it.
PROVIDER_ERRORS = (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)


class Setup(RunSetup[Row]):
    row_model = Row

    async def load_cases(self, row: Row, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        del runtime
        benchmark.task(row.input.domain, row.input.task_id)  # unknown task IDs fail here
        yield Case(id=row.id, input=row.input.model_dump(), expected={})


def chat_model(runtime: RolloutRuntime[Any]) -> ChatModel:
    """The run's selected model through Beaker's gateway; else the pinned default.

    A model choice resolves to a gateway endpoint, a run-scoped token and a
    canonical model name; the gateway applies the run's reasoning effort. Only
    the agent follows it: the user simulator's model is fixed in ``user_sim.py``.
    """
    if runtime.model:
        target = inference_target(runtime)
        client = openai.OpenAI(api_key=target.api_key, base_url=target.base_url, max_retries=2, timeout=300)
        return ChatModel(client, target.model)
    return benchmark.default_chat_model()


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime[Any]) -> CaseResult:
    ref = TaskRef.model_validate(case_input)
    # Repository candidates execute with source_dir as their working directory.
    instructions = load_instructions(Path.cwd() / "prompts")
    chat = chat_model(runtime)
    context: dict[str, JsonValue] = {"domain": ref.domain, "task_id": ref.task_id, "error": None}
    with runtime.trace.stage("tau2.run_task", inputs=ref.model_dump()) as stage:
        try:
            simulation = await asyncio.to_thread(
                benchmark.run_task,
                ref.domain,
                ref.task_id,
                lambda tools, policy: Agent(tools, policy, chat=chat, instructions=instructions),
                agent_model=chat.model,
            )
        except PROVIDER_ERRORS as exc:
            message = f"{type(exc).__name__}: {exc}"
            stage.output({"error": message})
            raise RetryableCaseError(message) from exc
        except Exception as exc:
            # An agent or simulator failure is a failed case with a visible error.
            context["error"] = f"{type(exc).__name__}: {exc}"
            stage.output({"error": context["error"]})
            return CaseResult(output={"error": context["error"]}, output_kind="record", context=context)
        finally:
            chat.client.close()
        output = scoring.summary(simulation)
        stage.output(output)
    return CaseResult(
        output=output, output_kind="record", context={**context, "simulation": scoring.to_context(simulation)}
    )


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    del case_files_dir
    ref = TaskRef.model_validate(case.input)
    basis = benchmark.reward_basis(ref.domain, ref.task_id)
    context = result.context
    if not context:
        # Beaker drops context when a result is over its size limit; that is not a failed case.
        raise RuntimeError("CaseResult.context is empty (dropped for size?); the trajectory cannot be scored")
    if context.get("error"):
        return scoring.error_score(str(context["error"]), basis)
    simulation = benchmark.load_simulation(context["simulation"])
    # Upstream's evaluator replays the trajectory (and calls the NL-assertion judge for retail).
    reward = await asyncio.to_thread(benchmark.evaluate, simulation, ref.domain, ref.task_id)
    return scoring.score(reward, simulation, basis)


integration = Integration(
    targets=repository(paths=("prompts", "src/tau2_bench/agent.py", "src/tau2_bench/llm.py")),
    run_setup=Setup,
    run_case=run_case,
    score_case=score_case,
)
