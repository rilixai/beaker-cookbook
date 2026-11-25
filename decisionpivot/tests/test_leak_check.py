"""leak-check fails on a prompt that quotes an item and passes on the committed prompts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from decisionpivot.data import SPLIT_FILES, find_leaks, load_rows


ROOT = Path(__file__).resolve().parents[1]


def run_leak_check(data: Path, *paths: Path) -> subprocess.CompletedProcess[str]:
    args = [a for p in paths for a in ("--path", str(p))]
    return subprocess.run(
        [sys.executable, "-m", "decisionpivot.cli", "leak-check", "--data", str(data), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_passes_on_committed_prompts_and_agent(prepared: Path) -> None:
    result = run_leak_check(prepared, ROOT / "prompts", ROOT / "src/decisionpivot/agent.py")
    assert result.returncode == 0, result.stdout + result.stderr


def test_fails_on_a_prompt_that_quotes_an_item(prepared: Path, tmp_path: Path) -> None:
    row = load_rows(prepared / SPLIT_FILES["heldout"])[0]
    state = row.input.items[0].state
    text = state if isinstance(state, str) else " ".join(str(v) for v in state.values())
    words = text.split()
    quote = " ".join(words[len(words) // 2 : len(words) // 2 + 14])
    prompt = tmp_path / "instructions.md"
    prompt.write_text(f"Answer carefully. For example:\n\n{quote.upper()}\n")  # case and line breaks do not hide it
    result = run_leak_check(prepared, prompt)
    assert result.returncode == 1
    assert '"leaks": ' in result.stdout and '"leaks": 0' not in result.stdout


def test_whole_short_state_and_general_guidance() -> None:
    import fake_api

    rows = [fake_api.row()]
    general = "Read every rule. Check the seat count against the size of the booking before you answer."
    assert find_leaks(rows, {"p": general}) == []
    eleven = "Room Kestrel seats 10 people. The booking is for 12 people"
    assert find_leaks(rows, {"p": eleven}) == []
    twelve = eleven + " at"
    assert [leak["item"] for leak in find_leaks(rows, {"p": twelve})] == ["sched_conflict:001a"]
