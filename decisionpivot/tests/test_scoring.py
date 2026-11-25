"""Group scoring, and run-level scores from the pinned upstream score.py on small hand-checked fixtures."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import fake_api

from decisionpivot.data import CaseInput, Expected, Item, Question, Row
from decisionpivot.scoring import score_group, top_option, upstream_scores, write_run


def output(row: Row, *ps: dict[str, float] | None) -> dict[str, Any]:
    return {
        "items": [
            {"id": i.id, "probabilities": p, "error": None if p is not None else "request_error: x"}
            for i, p in zip(row.input.items, ps, strict=True)
        ]
    }


def test_pair_correct_is_zero_when_one_member_is_wrong() -> None:
    row = fake_api.row()  # golds yes, no
    both = score_group(row, output(row, {"yes": 0.8, "no": 0.2}, {"yes": 0.3, "no": 0.7}))
    one = score_group(row, output(row, {"yes": 0.8, "no": 0.2}, {"yes": 0.6, "no": 0.4}))
    assert both.objective == 1.0 and both.field_scores["item_accuracy"] == 1.0
    assert one.objective == 0.0 and one.field_scores["pair_correct"] == 0.0
    assert one.field_scores["item_accuracy"] == 0.5
    assert math.isclose(one.field_scores["min_top_probability"], 0.6)
    assert one.field_scores["level3_pair_correct"] == 0.0 and both.field_scores["level3_pair_correct"] == 1.0
    assert [(s.expected, s.predicted) for s in one.items] == [("yes", "yes"), ("no", "yes")]


def test_tie_goes_to_the_first_listed_option_as_upstream() -> None:
    assert top_option(["no", "yes"], {"yes": 0.5, "no": 0.5}) == ("no", 0.5)
    row = fake_api.row(golds=("yes", "yes"))
    tied = score_group(row, output(row, {"yes": 0.5, "no": 0.5}, {"yes": 0.9, "no": 0.1}))
    assert [s.correct for s in tied.items] == [True, True]  # "yes" is listed first in the fixture


def test_failed_item_is_wrong_with_uniform_probability() -> None:
    row = fake_api.row()
    scored = score_group(row, output(row, None, {"yes": 0.1, "no": 0.9}))
    assert scored.objective == 0.0 and "level3_pair_correct" in scored.field_scores
    assert scored.items[0].predicted is None and scored.items[0].error
    assert scored.items[0].top_probability == 0.5
    missing = score_group(row, {"items": []})
    assert missing.objective == 0.0 and all(s.error == "no output" for s in missing.items)


YES_NO = Question(type="choice", instructions="?", criteria={"yes": "yes", "no": "no"})


def pairs(golds: list[tuple[str, str]], family: str = "state_goal", level: int = 1) -> list[Row]:
    rows = []
    for n, gs in enumerate(golds, 1):
        items = tuple(Item(id=f"{family}:{n:03d}{m}", state="s", questions={"q": YES_NO}) for m in "ab")
        rows.append(
            Row(
                id=f"dp-{family}:g{n:03d}",
                input=CaseInput(family=family, items=items),
                expected=Expected(gold={i.id: g for i, g in zip(items, gs, strict=True)}, level=level, layer="D"),
            )
        )
    return rows


def run(tmp_path: Path, upstream: Path, rows: list[Row], answers: list[list[dict[str, float] | None]]) -> Any:
    outputs = {r.id: output(r, *a) for r, a in zip(rows, answers, strict=True)}
    write_run(tmp_path, rows, outputs, upstream)
    families = sorted({r.input.family for r in rows})
    return upstream_scores(tmp_path, upstream, families, resamples=20)


def py(p: float) -> dict[str, float]:
    return {"yes": p, "no": 1 - p}


def test_upstream_schema_examples(tmp_path: Path, upstream: Path) -> None:
    # docs/SCHEMA.md: two pairs, both members right in the first and one in the second -> accuracy 75, pair 50.
    # Pair direction: P(yes) 0.8 against 0.3 counts, 0.55 against 0.6 does not -> 50.
    rows = pairs([("yes", "no"), ("yes", "no")])
    answers = [[py(0.8), py(0.3)], [py(0.55), py(0.6)]]
    scores = run(tmp_path, upstream, rows, answers)
    task = scores["tasks"]["state_goal"]
    assert task["accuracy"] == 0.75 and task["pair_accuracy"] == 0.5 and task["pair_direction"] == 0.5
    ours = [score_group(r, output(r, *a)).objective for r, a in zip(rows, answers, strict=True)]
    assert sum(ours) / len(ours) == task["pair_accuracy"]


def test_upstream_unanswered_and_auto_at_5(tmp_path: Path, upstream: Path) -> None:
    # tests/test_score.py and docs/SCHEMA.md: ten items, the eight most confident right, the ninth wrong
    # (prefix of nine has 11% errors) -> auto_at_5 = 80.  The tenth is unanswered: uniform and wrong.
    rows = pairs([("yes", "no")] * 5)
    confident = [0.99, 0.01, 0.98, 0.02, 0.97, 0.03, 0.96, 0.04]
    answers = [[py(confident[0]), py(confident[1])], [py(confident[2]), py(confident[3])]]
    answers += [[py(confident[4]), py(confident[5])], [py(confident[6]), py(confident[7])], [py(0.3), None]]
    task = run(tmp_path, upstream, rows, answers)["tasks"]["state_goal"]
    assert math.isclose(task["auto_at_5"], 0.8)
    assert math.isclose(task["accuracy"], 0.8) and math.isclose(task["pair_accuracy"], 0.8)
    assert math.isclose(task["coverage"], 0.9)
    assert math.isclose(task["share_at_090"], 0.8) and task["acc_at_090"] == 1.0


def test_upstream_level3_pair_accuracy_and_core_mean(tmp_path: Path, upstream: Path) -> None:
    rows = pairs([("yes", "no")], level=1) + pairs([("yes", "no")], family="toolcall_valid", level=3)
    scores = run(tmp_path, upstream, rows, [[py(0.9), py(0.1)], [py(0.9), py(0.9)]])
    assert scores["tasks"]["toolcall_valid"]["levels"]["3"]["pair_accuracy"] == 0.0
    assert scores["core"]["families"] == 2 and scores["core"]["pair_accuracy"] == 0.5
