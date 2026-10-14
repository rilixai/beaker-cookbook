"""Capture native assertion execution without changing verdicts or app rules."""

from __future__ import annotations

import json
import linecache
import sys
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from inspect import CO_ASYNC_GENERATOR, CO_GENERATOR
from textwrap import dedent
from types import FrameType
from typing import Any


def _brief(value: Any, depth: int = 0) -> Any:
    if isinstance(value, str):
        return value if len(value) <= 400 else value[:400] + "… [truncated]"
    if value is None or isinstance(value, bool | int | float):
        return value
    if depth < 2:
        if isinstance(value, Mapping):
            items = list(value.items())[:8]
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
                            "file": frame.f_code.co_filename,
                            "line": frame.f_lineno,
                            "function": frame.f_code.co_name,
                            "returned": result,
                            "values": values,
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


def failure_trace(assertion: Mapping[str, Any], events: Iterable[Mapping[str, Any]]) -> str:
    parts = [f"Condition {assertion['type']} evaluated to False."]
    for event in events:
        source = dedent(
            "".join(
                linecache.getline(event["file"], line) for line in range(max(1, event["line"] - 2), event["line"] + 1)
            )
        ).strip()
        parts.append(
            f"{event['function']} returned {event['returned']}\n"
            f"```python\n{source}\n```\n"
            "Local values (bounded execution snapshot):\n"
            + json.dumps(event["values"], ensure_ascii=False, indent=2, default=str)
        )
    text = "\n\n".join(parts)
    encoded = text.encode("utf-8")
    return (
        text
        if len(encoded) <= 6_000
        else encoded[:5_960].decode("utf-8", errors="ignore") + "\n… diagnostic truncated"
    )
