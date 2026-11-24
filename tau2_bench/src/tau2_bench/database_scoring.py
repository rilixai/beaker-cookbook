"""Partial credit for reference state changes, independent of the tool sequence."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment


_MISSING = object()


class ReferenceReplayError(ValueError):
    """The reference cannot define a valid database partial-credit target."""


def compare_states(
    initial: dict[str, Any], expected: dict[str, Any], predicted: dict[str, Any]
) -> list[dict[str, Any]]:
    """Score changed record fields, plus unexpected changes; ignore untouched data.

    Lists of records (e.g. ordered items) are compared entry by
    entry. Other lists, nested fields such as addresses, and new/deleted records
    are atomic. This preserves related values without counting every scalar.
    """
    outcomes: list[dict[str, Any]] = []

    def add(path: list[str], before: Any, target: Any, actual: Any) -> None:
        if before == target and actual == target:
            return
        outcomes.append(
            {
                "name": "database outcome: " + "/".join(path),
                "passed": actual == target,
                "expected": None if target is _MISSING else target,
                "predicted": None if actual is _MISSING else actual,
                "message": "Unexpected database change." if before == target else "Required final database value.",
            }
        )

    def walk(path: list[str], before: Any, target: Any, actual: Any) -> None:
        # Define units from trusted initial/reference states, never candidate shape.
        # side / table / record / field: fields are the comparison units.
        if len(path) < 4 and isinstance(before, dict) and isinstance(target, dict):
            observed = actual if isinstance(actual, dict) else {}
            keys = before.keys() | target.keys() | observed.keys()
            if not keys:
                add(path, before, target, actual)
            for key in sorted(keys):
                walk(path + [key], before.get(key, _MISSING), target.get(key, _MISSING), observed.get(key, _MISSING))
        elif (
            len(path) == 4
            and isinstance(before, list)
            and isinstance(target, list)
            and all(isinstance(item, dict) for values in (before, target) for item in values)
        ):
            entries = actual if isinstance(actual, list) else []
            for index in range(max(len(before), len(target), len(entries))):
                add(
                    path + [str(index)],
                    before[index] if index < len(before) else _MISSING,
                    target[index] if index < len(target) else _MISSING,
                    entries[index] if index < len(entries) else _MISSING,
                )
            if not before and not target and not entries:
                add(path, before, target, actual)
        else:
            add(path, before, target, actual)

    walk([], initial, expected, predicted)
    return outcomes


def _snapshot(environment: Environment) -> dict[str, Any]:
    return {
        side: deepcopy(toolkit.db.model_dump()) if toolkit is not None and toolkit.db is not None else {}
        for side, toolkit in (("agent", environment.tools), ("user", environment.user_tools))
    }


def database_outcomes(
    simulation: SimulationRun,
    task: Task,
    environment_constructor: Callable[..., Environment],
    env_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Replay the same initial, reference and predicted worlds as tau2's DB check.

    Failed reference writes cannot define partial-credit targets. Read-only
    reference calls do not change the target and need not execute. These worlds
    stay in evaluator memory, outside the candidate's result payload.
    """
    state = task.initial_state
    initialization = {
        "initialization_data": state.initialization_data if state else None,
        "initialization_actions": state.initialization_actions if state else None,
    }
    gold = environment_constructor(**deepcopy(env_kwargs))
    gold.set_state(**initialization, message_history=state.message_history or [] if state else [])
    initial = _snapshot(gold)
    for action in task.evaluation_criteria.actions or []:
        if not gold._is_mutating_tool(action.name):
            continue
        try:
            gold.make_tool_call(tool_name=action.name, requestor=action.requestor, **action.arguments)
        except Exception as exc:
            raise ReferenceReplayError(f"Reference {action.name} failed: {exc}") from exc
    expected = _snapshot(gold)
    predicted = environment_constructor(**deepcopy(env_kwargs))
    predicted.set_state(**initialization, message_history=list(simulation.messages))
    return compare_states(initial, expected, _snapshot(predicted))
