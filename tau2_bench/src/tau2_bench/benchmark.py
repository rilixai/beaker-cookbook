"""Run one tau2 task as a full conversation, and score it with tau2's own evaluator.

Fixed benchmark code, outside Beaker's editable set. Tasks come from the
English language binding (as every upstream 2.0 run does), so retail uses its
50 reviewed task replacements.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from contextvars import ContextVar
from functools import cache
from typing import Any

from loguru import logger
from openai import OpenAI
from tau2.agent.base_agent import HalfDuplexAgent
from tau2.data_model.semantic_communication import CommunicateScorer, CommunicateScoring
from tau2.data_model.simulation import RewardInfo, SimulationRun, TextRunConfig
from tau2.data_model.tasks import RewardType, Task
from tau2.environment.tool import Tool
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.multilingual.runtime import (
    language_environment_kwargs,
    pin_language_artifacts,
    reload_call_language,
    task_context,
)
from tau2.registry import registry
from tau2.runner.build import build_text_orchestrator
from tau2.utils.utils import DATA_DIR as tau2_data_dir

from tau2_bench.data import Domain, data_dir, ensure_data
from tau2_bench.llm import ChatModel
from tau2_bench.user_sim import USER_LLM, USER_LLM_ARGS, USER_NAME, TracedUserSimulator


AGENT_NAME = "beaker_tau2_agent"
# Pinned in fixed code so that editing the agent never changes what a default run evaluates.
DEFAULT_AGENT_MODEL = "gpt-5.6-luna"
DEFAULT_AGENT_REASONING = "medium"
# Deterministic communicate scoring (the CLI's ``--communicate-scorer substring``).
COMMUNICATE_SCORING = CommunicateScoring(scorer=CommunicateScorer.SUBSTRING)

AgentBuilder = Callable[[list[Tool], str], HalfDuplexAgent[Any]]
_agent_builder: ContextVar[AgentBuilder] = ContextVar("tau2_bench_agent_builder")


def _agent_factory(*, tools: list[Tool], domain_policy: str, **_: Any) -> HalfDuplexAgent[Any]:
    # tau2 builds agents through a global registry; the per-case builder rides
    # in a context variable so concurrent cases each get their own agent.
    return _agent_builder.get()(tools, domain_policy)


@cache
def _setup() -> None:
    if tau2_data_dir.resolve() != data_dir().resolve():
        raise RuntimeError("tau2 was imported before tau2_bench; import tau2_bench first so TAU2_DATA_DIR is set.")
    ensure_data()
    logger.remove()
    logger.add(sys.stderr, level="ERROR")  # upstream default; per-message cost warnings are noise
    registry.register_agent_factory(_agent_factory, AGENT_NAME)
    registry.register_user(TracedUserSimulator, USER_NAME)


def run_config(domain: Domain, agent_model: str = DEFAULT_AGENT_MODEL) -> TextRunConfig:
    _setup()
    return TextRunConfig(
        domain=domain,
        agent=AGENT_NAME,
        llm_agent=agent_model,  # recorded only; the agent builds its own client
        user=USER_NAME,
        llm_user=USER_LLM,
        llm_args_user=dict(USER_LLM_ARGS),
        communicate_scoring=COMMUNICATE_SCORING,
    )


@cache
def _tasks(domain: Domain) -> dict[str, Task]:
    loaded = pin_language_artifacts(run_config(domain), update=False)
    return {task.id: task for task in loaded.tasks}


def task_ids(domain: Domain) -> list[str]:
    return list(_tasks(domain))


def task(domain: Domain, task_id: str) -> Task:
    found = _tasks(domain).get(task_id)
    if found is None:
        raise KeyError(f"unknown tau2 {domain} task_id {task_id!r}")
    return found


def default_chat_model() -> ChatModel:
    """The agent model a run without a model choice uses: ``OPENAI_API_KEY`` (and ``OPENAI_BASE_URL``)."""
    return ChatModel(OpenAI(max_retries=2, timeout=300), DEFAULT_AGENT_MODEL, DEFAULT_AGENT_REASONING)


def reward_basis(domain: Domain, task_id: str) -> list[RewardType]:
    criteria = task(domain, task_id).evaluation_criteria
    return list(criteria.reward_basis) if criteria is not None else []


def run_task(domain: Domain, task_id: str, build_agent: AgentBuilder, *, agent_model: str) -> SimulationRun:
    """Run the conversation to its end, without scoring. Upstream's turn cap applies."""
    selected = task(domain, task_id)
    token = _agent_builder.set(build_agent)
    try:
        orchestrator = build_text_orchestrator(run_config(domain, agent_model), selected)
        return orchestrator.run()
    finally:
        _agent_builder.reset(token)


def load_simulation(recorded: Any) -> SimulationRun:
    return SimulationRun.model_validate(recorded)


def evaluate(simulation: SimulationRun, domain: Domain, task_id: str) -> RewardInfo:
    """tau2's reward: the product of the task's ``reward_basis`` components.

    Airline is DB x COMMUNICATE (substring match). Retail is DB x NL_ASSERTION,
    judged by upstream's pinned NL-assertion model.
    """
    config = run_config(domain)
    selected = task(domain, task_id)
    context = task_context(config, selected)
    env_kwargs = language_environment_kwargs(reload_call_language(context, domain)) or None
    return evaluate_simulation(
        simulation=simulation,
        task=selected,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain=domain,
        env_kwargs=env_kwargs,
        communicate_scoring=COMMUNICATE_SCORING,
        customer_language=context.provider_language,
    )
