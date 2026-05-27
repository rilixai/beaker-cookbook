"""Shared gold-supporting-fact helpers for the HotpotQA agent.

The PydanticAI agent (under ``.agent``) and its feedback module both need
to:

* compute which gold titles are still unretrieved at a given point in
  the trajectory, comparing case-insensitively;
* assemble the artifact-style "ideal summary" string from
  ``record.supporting_titles × record.supporting_sentence_ids``.

Centralized here so any future bug fix lands in one place.
"""

from __future__ import annotations

from collections.abc import Sequence

from .dataset import HotpotQARecord


def remaining_gold_titles(
    gold_titles: Sequence[str],
    retrieved_titles_seen: set[str],
) -> list[str]:
    """Return gold titles not yet retrieved, comparing case-insensitively.

    Both arguments are normalized via ``str.strip().lower()`` for the
    membership check, but the returned list preserves the original
    casing from ``gold_titles`` (which is what the reflection feedback
    and ``HopTrace`` bookkeeping want to display).
    """
    seen_lower = {t.strip().lower() for t in retrieved_titles_seen}
    return [title for title in gold_titles if title.strip().lower() not in seen_lower]


def ideal_summary_from_supporting_facts(record: HotpotQARecord) -> str:
    """Build the artifact's "ideal summary" string for a HotpotQA record.

    The artifact assembles it from ``zip(supporting_facts.title,
    supporting_facts.sent_id)`` — one ``"title | sentence"`` line per
    gold supporting fact, joined with ``"\\n   "`` (three-space indent
    after each newline so the result fits inside an indented block of
    reflection feedback).
    """
    title_to_sentences = {p.title: p.sentences for p in record.paragraphs}
    lines: list[str] = []
    for title in record.supporting_titles:
        sentences = title_to_sentences.get(title, ())
        sent_ids = record.supporting_sentence_ids.get(title, ())
        for sent_id in sent_ids:
            if 0 <= sent_id < len(sentences):
                lines.append(f"{title} | {sentences[sent_id]}")
    return "\n   ".join(lines)


__all__ = ["ideal_summary_from_supporting_facts", "remaining_gold_titles"]
