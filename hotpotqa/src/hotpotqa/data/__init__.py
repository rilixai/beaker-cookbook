"""Raw HotpotQA primitives — no GEPA dependency.

The "ground truth" shelf: dataset loading, the official scorer
primitives, and supporting-fact utilities. None of these modules
import from rilixai. A reader writing a new benchmark looks here to
see what the underlying dataset surface looks like, then to
:mod:`hotpotqa.optimization` to see how it gets wired into GEPA.
"""

from __future__ import annotations

from .dataset import (
    HotpotQAParagraph,
    HotpotQARecord,
    cases_from_records,
    load_hotpotqa_paper_split,
    load_hotpotqa_split,
    record_to_case,
)
from .eval import exact_match_score, f1_score, f1_score_components, normalize_answer
from .gold import ideal_summary_from_supporting_facts, remaining_gold_titles


__all__ = [
    "HotpotQAParagraph",
    "HotpotQARecord",
    "cases_from_records",
    "exact_match_score",
    "f1_score",
    "f1_score_components",
    "ideal_summary_from_supporting_facts",
    "load_hotpotqa_paper_split",
    "load_hotpotqa_split",
    "normalize_answer",
    "record_to_case",
    "remaining_gold_titles",
]
