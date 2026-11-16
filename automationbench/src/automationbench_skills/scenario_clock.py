"""Resolve the clock for published tasks without consulting the host clock."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, time
from typing import Any


DEFAULT_SCENARIO_TIME = "2026-02-20T12:00:00+00:00"
_MONTHS = {
    name: index
    for index, names in enumerate(
        (
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ),
        start=1,
    )
    for name in names
}
_CURRENT_DATE = re.compile(
    r"\b(?:today(?:['’]s date)?\s*(?:is\s+|:\s*)?\(?\s*"
    r"|current (?:date(?: and time)?|time)\s*(?:is|:)\s*)"
    r"(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+)?"
    r"(?P<value>\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2}))?"
    r"|(?:" + "|".join(_MONTHS) + r")\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)


def scenario_time(prompt: list[dict[str, Any]], initial_state: dict[str, Any]) -> str:
    """Use the world clock, then an explicit current date in the task, then a fixed fallback.

    Only current-date statements count; event dates and deadlines do not. Date-only
    statements use noon UTC, as does the fallback. Frozen fixtures retain their
    saved scenario fallback and do not use this resolver.
    """
    authored = (initial_state.get("meta") or {}).get("current_time")
    if isinstance(authored, str) and authored:
        return authored
    dates: set[date] = set()
    timestamps: set[datetime] = set()
    for message in prompt:
        if message.get("role") != "user":
            continue
        for match in _CURRENT_DATE.finditer(str(message.get("content") or "")):
            value = match["value"]
            if value[0].isdigit() and "T" in value.upper():
                stamp = datetime.fromisoformat(value.upper())
                timestamps.add(stamp)
                dates.add(stamp.date())
            elif value[0].isdigit():
                dates.add(date.fromisoformat(value))
            else:
                month, day, year = value.replace(",", "").lower().split()
                dates.add(date(int(year), _MONTHS[month], int(day)))
    if len(dates) > 1 or len(timestamps) > 1:
        raise ValueError("Task contains conflicting current-date statements")
    if timestamps:
        return next(iter(timestamps)).isoformat()
    if dates:
        return datetime.combine(next(iter(dates)), time(12), tzinfo=UTC).isoformat()
    return DEFAULT_SCENARIO_TIME
