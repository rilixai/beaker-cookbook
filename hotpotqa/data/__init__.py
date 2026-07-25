"""Raw HotpotQA primitives.

The "ground truth" shelf: dataset loading, the official scorer
primitives, and supporting-fact utilities. A reader wiring this agent
to a new task looks here to see what the underlying dataset surface
looks like, then to :mod:`hotpotqa.evaluation` to see how the agent's
answers get scored against it.
"""

from __future__ import annotations

from .dataset import (
    HotpotQAParagraph,
    HotpotQARecord,
    load_hotpotqa_paper_split,
    load_hotpotqa_split,
    records_from_raw,
)
from .eval import exact_match_score, f1_score, f1_score_components, normalize_answer
from .gold import remaining_gold_titles


__all__ = [
    "HotpotQAParagraph",
    "HotpotQARecord",
    "exact_match_score",
    "f1_score",
    "f1_score_components",
    "load_hotpotqa_paper_split",
    "load_hotpotqa_split",
    "normalize_answer",
    "records_from_raw",
    "remaining_gold_titles",
]
