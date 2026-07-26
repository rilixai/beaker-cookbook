"""Shared gold-supporting-fact helpers for the HotpotQA agent.

The agent tracks which gold titles it has already retrieved so its
retrieval trace can be inspected hop by hop; the comparison is
case-insensitive. Centralized here so any future bug fix lands in one
place.
"""

from __future__ import annotations

from collections.abc import Sequence


def remaining_gold_titles(
    gold_titles: Sequence[str],
    retrieved_titles_seen: set[str],
) -> list[str]:
    """Return gold titles not yet retrieved, comparing case-insensitively.

    Both arguments are normalized via ``str.strip().lower()`` for the
    membership check, but the returned list preserves the original
    casing from ``gold_titles`` (which is what the retrieval-trace
    bookkeeping wants to display).
    """
    seen_lower = {t.strip().lower() for t in retrieved_titles_seen}
    return [title for title in gold_titles if title.strip().lower() not in seen_lower]


__all__ = ["remaining_gold_titles"]
