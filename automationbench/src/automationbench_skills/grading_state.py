"""Transport PI's transient grading evidence alongside the serialized world."""

from __future__ import annotations

from collections.abc import Mapping

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
