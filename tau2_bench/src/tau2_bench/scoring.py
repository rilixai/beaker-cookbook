"""Combine partial and total correctness; retain tau2 reward for comparison. Fixed code."""

from __future__ import annotations

import json
from typing import Any, Literal

from beaker import CaseScore, Check, JsonValue
from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
from tau2.data_model.tasks import RewardType
from tau2.utils.llm_utils import get_cost


FIELD_NAMES = {
    RewardType.DB: "db",
    RewardType.COMMUNICATE: "communicate",
    RewardType.NL_ASSERTION: "nl_assertion",
    RewardType.ENV_ASSERTION: "env_assertion",
    RewardType.ACTION: "action",
}
_COMPLETED = {TerminationReason.AGENT_STOP, TerminationReason.USER_STOP}
_TEXT_LIMIT = 500
PARTIAL_WEIGHT = 0.7
TOTAL_WEIGHT = 0.3


def to_context(simulation: SimulationRun) -> dict[str, JsonValue]:
    """The trajectory the evaluator needs, without raw provider responses.

    The final DB state is not stored: the evaluator rebuilds it by replaying
    the trajectory, and the airline/retail databases exceed the result limit.
    """
    dumped: dict[str, JsonValue] = simulation.model_dump(
        mode="json", exclude={"messages": {"__all__": {"raw_data"}}, "reward_info": True}
    )
    return dumped


def summary(simulation: SimulationRun) -> dict[str, JsonValue]:
    """Small display output: how the conversation ended and what the agent did."""
    assistant = [m for m in simulation.messages if isinstance(m, AssistantMessage)]
    tool_calls = [call.name for message in assistant for call in message.tool_calls or []]
    final = next((m.content for m in reversed(assistant) if m.content), None)
    tokens = {"prompt_tokens": 0, "completion_tokens": 0}
    for message in assistant:
        if message.usage:
            for key in tokens:
                tokens[key] += int(message.usage.get(key) or 0)
    _, user_cost = get_cost(simulation.messages)
    return {
        "termination_reason": simulation.termination_reason.value,
        "messages": len(simulation.messages),
        "agent_turns": len(assistant),
        "agent_tokens": dict(tokens),
        "user_sim_cost_usd": user_cost,
        "tool_calls": tool_calls[:50],
        "final_agent_message": final[:_TEXT_LIMIT] if final else None,
    }


def _verdict(passed: bool) -> Literal["pass", "fail"]:
    return "pass" if passed else "fail"


def error_score(error: str, basis: list[RewardType]) -> CaseScore:
    """An agent or simulator failure: a failed case with the error shown, never a pass."""
    return CaseScore(
        objective=0.0,
        field_scores={
            "reward": 0.0,
            "benchmark_reward": 0.0,
            "partial_correctness": 0.0,
            "total_correctness": 0.0,
            "combined_correctness": 0.0,
            **{FIELD_NAMES[b]: 0.0 for b in basis},
        },
        checks=(Check(name="conversation ran", verdict="fail", message=error),),
    )


def score(reward: RewardInfo, simulation: SimulationRun, basis: list[RewardType]) -> CaseScore:
    completed = simulation.termination_reason in _COMPLETED
    breakdown: dict[Any, float] = reward.reward_breakdown or {}
    # A premature end scores 0 upstream without computing components; report
    # them as 0 too so per-component means are not computed over finished runs only.
    fields = {FIELD_NAMES[b]: float(breakdown.get(b, 0.0)) if completed else 0.0 for b in basis}
    checks: list[Check] = [
        Check(
            name="conversation ended normally",
            verdict=_verdict(completed),
            predicted=simulation.termination_reason.value,
            message=None if completed else str((reward.info or {}).get("note") or ""),
        )
    ]
    database_outcomes = (reward.info or {}).get("database_outcomes")
    if reward.db_check is not None:
        checks.append(
            Check(
                name="database matches the expected end state",
                verdict=_verdict(reward.db_check.db_match),
                group="DB",
                informational=RewardType.DB not in basis or database_outcomes is not None,
            )
        )
    for outcome in database_outcomes or []:
        checks.append(
            Check(
                name=outcome["name"],
                verdict=_verdict(outcome["passed"]),
                expected=outcome["expected"],
                predicted=outcome["predicted"],
                message=outcome["message"],
                group="DB_OUTCOME",
                informational=RewardType.DB not in basis,
            )
        )
    if reference_error := (reward.info or {}).get("database_outcome_error"):
        checks.append(
            Check(
                name="database partial-credit reference replay",
                verdict="fail",
                group="REFERENCE",
                message=f"{reference_error}. Using the original database check for partial credit.",
                informational=True,
            )
        )
    for comm in reward.communicate_checks or []:
        checks.append(
            Check(
                name=f"communicates: {comm.info}",
                verdict=_verdict(comm.met),
                expected=comm.info,
                message=comm.justification or None,
                group="COMMUNICATE",
                informational=RewardType.COMMUNICATE not in basis,
            )
        )
    for nl in reward.nl_assertions or []:
        checks.append(
            Check(
                name=nl.nl_assertion,
                verdict=_verdict(nl.met),
                message=nl.justification or None,
                group="NL_ASSERTION",
                informational=RewardType.NL_ASSERTION not in basis,
            )
        )
    for action in reward.action_checks or []:
        # Gold actions are reported for diagnosis only; they are not in airline/retail reward bases.
        checks.append(
            Check(
                name=f"{action.action.name}({json.dumps(action.action.arguments, sort_keys=True)})",
                verdict=_verdict(action.action_match),
                group="ACTION",
                informational=RewardType.ACTION not in basis,
            )
        )
    benchmark_reward = float(reward.reward) if completed else 0.0
    # The normal-termination row is diagnostic, not free partial credit.
    scored_checks = [check for check in checks[1:] if not check.informational]
    partial = benchmark_reward
    if completed and scored_checks:
        partial = sum(check.verdict == "pass" for check in scored_checks) / len(scored_checks)
    total = float(completed and benchmark_reward == 1.0)
    objective = PARTIAL_WEIGHT * partial + TOTAL_WEIGHT * total
    return CaseScore(
        objective=objective,
        field_scores={
            "reward": benchmark_reward,
            "benchmark_reward": benchmark_reward,
            "partial_correctness": partial,
            "total_correctness": total,
            "combined_correctness": objective,
            **fields,
        },
        checks=tuple(checks),
    )
