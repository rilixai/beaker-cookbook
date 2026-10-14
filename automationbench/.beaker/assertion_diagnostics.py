"""Capture native assertion execution without changing verdicts or app rules."""

from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from inspect import CO_ASYNC_GENERATOR, CO_GENERATOR
from itertools import islice
from types import FrameType
from typing import Any


def _brief(value: Any, depth: int = 0) -> Any:
    if isinstance(value, str):
        return value if len(value) <= 400 else value[:400] + "… [truncated]"
    if value is None or isinstance(value, bool | int | float):
        return value
    if depth < 2:
        if isinstance(value, Mapping):
            items = list(islice(value.items(), 8))
            bounded_map = {str(key): _brief(child, depth + 1) for key, child in items}
            if len(value) > 8:
                bounded_map["…"] = f"{len(value) - 8} more entries"
            return bounded_map
        if isinstance(value, list | tuple | set | frozenset):
            sequence = sorted(value, key=str) if isinstance(value, set | frozenset) else value
            bounded_items = [_brief(child, depth + 1) for child in sequence[:8]]
            if len(value) > 8:
                bounded_items.append(f"… {len(value) - 8} more entries")
            return bounded_items
    return f"<{type(value).__name__}>"


@contextmanager
def capture_assertion_returns(
    world: Any, assertions: list[dict[str, Any]]
) -> Iterator[dict[int, deque[dict[str, Any]]]]:
    """Profile only this synchronous rubric call and its end-state assertions.

    Initial-state checks are ignored. Aliases/helpers are attributed through
    their calling assertion. Keep the last three boolean returns as execution
    snapshots, not a guessed explanation of which predicate caused failure.
    Leave existing profilers alone, and always restore the previous state.
    """
    captured: dict[int, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=3))
    identities = {id(assertion) for assertion in assertions}
    previous = sys.getprofile()

    def profile(frame: FrameType, event: str, result: Any) -> None:
        if event != "return" or not isinstance(result, bool):
            return
        if not str(frame.f_globals.get("__name__", "")).startswith("automationbench.rubric.assertions."):
            return
        if frame.f_code.co_flags & (CO_GENERATOR | CO_ASYNC_GENERATOR):
            return
        try:
            owner: FrameType | None = frame
            while owner is not None:
                identity = id(owner.f_locals.get("assertion"))
                if (
                    identity in identities
                    and "world" in owner.f_locals
                    and str(owner.f_globals.get("__name__", "")).startswith("automationbench.rubric.")
                ):
                    if owner.f_locals["world"] is not world:
                        return
                    values = {
                        name: _brief(value)
                        for name, value in frame.f_locals.items()
                        if name not in {"world", "assertion"}
                        and isinstance(
                            value, (str, bool, int, float, list, tuple, set, frozenset, Mapping, type(None))
                        )
                    }
                    captured[identity].append(
                        {
                            "function": frame.f_code.co_name,
                            "returned": result,
                            "values": values,
                            "collection_sizes": {
                                name: len(value)
                                for name, value in frame.f_locals.items()
                                if name not in {"world", "assertion"}
                                and isinstance(value, (list, tuple, set, frozenset, Mapping))
                            },
                        }
                    )
                    break
                owner = owner.f_back
        except Exception:
            # Diagnostics must never change the rubric's behavior.
            return

    try:
        if previous is None:
            sys.setprofile(profile)
        yield captured
    finally:
        if previous is None:
            sys.setprofile(previous)


def _label(name: str) -> str:
    return {"col": "Column", "expected_val": "Required value", "cell_value": "Observed value"}.get(
        name, name.replace("_", " ").capitalize()
    )


def _display(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def failure_trace(assertion: Mapping[str, Any], events: Iterable[Mapping[str, Any]]) -> str:
    """Render requirements and recorded evidence without guessing a failure cause."""
    snapshots = list(events)
    requirements = [
        f"- {_label(name)}: {_display(_brief(value))}"
        for name, value in assertion.items()
        if name not in {"type", "scored", "excluded"}
    ]
    parts = ["Required conditions:\n" + "\n".join(requirements)] if requirements else []
    comparisons = []
    for event in snapshots:
        # These shared helpers compare values across apps. A helper match can
        # also violate a negative assertion; do not label it a Check pass/fail.
        function = event["function"]
        if function != "_cell_values_equal" and not function.startswith("_contains_"):
            continue
        values = event["values"]
        for observed, required in (("actual", "expected"), ("haystack", "needle"), ("hay", "ndl")):
            if observed in values and required in values:
                comparisons.append(
                    f"- {_display(values[observed])} compared with {_display(values[required])}: "
                    + ("match" if event["returned"] else "no match")
                )
                break
    if comparisons:
        parts.append("Observed comparisons (values may be normalized):\n" + "\n".join(comparisons))
    if snapshots:
        latest = snapshots[-1]
        observations = []
        for name, value in latest["values"].items():
            if value is None or name in {"prefix", "suffix", "match"}:
                continue
            if name in assertion and value == _brief(assertion[name]):
                continue
            count = latest.get("collection_sizes", {}).get(name)
            if (
                isinstance(value, list)
                and value
                and all(
                    isinstance(item, str) and ((item.startswith("<") and item.endswith(">")) or item.startswith("… "))
                    for item in value
                )
            ):
                rendered = f"{count} items (contents not captured)" if count is not None else "contents not captured"
            else:
                rendered = _display(value)
            observations.append(f"- {_label(name)}: {rendered}")
        if observations:
            parts.append("Latest evaluator observations:\n" + "\n".join(observations))
    else:
        parts.append("Detailed execution evidence was not captured; see OUTPUT for observed app changes.")
    text = "\n\n".join(parts)
    limit_notice = (
        "\n\nPartial diagnostic: the comparisons shown are not exhaustive and do not by themselves explain "
        "the overall failure. Long values and collections may be shortened."
        if snapshots
        else ""
    )
    encoded = text.encode("utf-8")
    budget = 6_000 - len(limit_notice.encode("utf-8"))
    if len(encoded) > budget:
        marker = "\n… diagnostic truncated"
        text = encoded[: budget - len(marker.encode("utf-8"))].decode("utf-8", errors="ignore") + marker
    return text + limit_notice
