"""Hermetic fixtures: a synthetic Pro CSV, a fake corpus cache, a scripted model client."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from officeqa import config
from officeqa.agent.agent import LLMResponse, Usage
from officeqa.data.corpus import Workspace, create_workspace, materialize
from officeqa.data.dataset import EvalRecord, load_records_from_csv


# Synthetic rows only: no benchmark content is redistributed in this repo.
ROWS = [
    {
        "uid": "q-0001",
        "question": "What were total receipts (in millions) in fiscal year 1987 per the June 1987 bulletin?",
        "answer": "9876543.21",
        "source_docs": "Treasury Bulletin June 1987",
        "source_files": "treasury_bulletin_1987_06.txt",
        "difficulty": "hard",
    },
    {
        "uid": "q-0002",
        "question": "Which state had the highest ratio in March 2013?",
        "answer": "[Massachusetts, 0.866]",
        "source_docs": "Treasury Bulletin March 2013",
        "source_files": "treasury_bulletin_2013_03.txt\ntreasury_bulletin_2013_06.txt",
        "difficulty": "hard",
    },
    {
        "uid": "q-0003",
        "question": "How much did the figure change between 1941 and 1943?",
        "answer": "$7,046,001.98",
        "source_docs": "Treasury Bulletin",
        "source_files": "treasury_bulletin_1941_01.txt\ntreasury_bulletin_1942_01.txt\ntreasury_bulletin_1943_01.txt",
        "difficulty": "hard",
    },
]


@pytest.fixture
def mini_csv(tmp_path: Path) -> Path:
    path = tmp_path / "mini_pro.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ROWS[0]))
        w.writeheader()
        w.writerows(ROWS)
    return path


@pytest.fixture
def records(mini_csv: Path) -> list[EvalRecord]:
    return load_records_from_csv(mini_csv)


FAKE_DOCS = {
    "treasury_bulletin_1987_06": "Treasury Bulletin June 1987\n\n| Item | Amount |\n| --- | --- |\n| Total receipts | 9876543.21 |\n"
    + "\n".join(f"filler line {i} national saving" for i in range(300)),
    "treasury_bulletin_2013_03": "Treasury Bulletin March 2013\n\n| State | Ratio |\n| --- | --- |\n| Massachusetts | 0.866 |\n| Texas | 0.512 |\n",
    "treasury_bulletin_2013_06": "Treasury Bulletin June 2013\n\nIndividual income taxes rose.\n",
}


@pytest.fixture
def fake_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A cache dir with a tiny fake corpus in both representations, wired via OFFICEQA_CACHE_DIR."""
    cache = tmp_path / "cache"
    pdf_dir = cache / "hf" / config.CORPUS_REPRESENTATIONS["pdfs"][0]
    txt_dir = cache / "hf" / config.CORPUS_REPRESENTATIONS["parsed"][0]
    pdf_dir.mkdir(parents=True)
    txt_dir.mkdir(parents=True)
    for name, text in FAKE_DOCS.items():
        (txt_dir / f"{name}.txt").write_text(text, encoding="utf-8")
        (pdf_dir / f"{name}.pdf").write_bytes(b"%PDF-1.4\n\x00\x00binary" + text.encode() + b"\x00%%EOF")
    monkeypatch.setenv("OFFICEQA_CACHE_DIR", str(cache))
    materialize(cache)
    return cache


@pytest.fixture
def corpus(fake_cache: Path) -> Path:
    return fake_cache / config.CORPUS_ROOT_NAME


@pytest.fixture
def workspace(tmp_path: Path, corpus: Path) -> Workspace:
    return create_workspace(tmp_path / "work", "q-0001", corpus, isolate=False)


# --- Scripted model client ---------------------------------------------------

Turn = LLMResponse | Callable[[list[dict[str, Any]]], LLMResponse]


def tool_call(name: str, call_id: str = "call_1", **arguments: Any) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        ],
        usage=Usage(prompt_tokens=100, completion_tokens=10),
        cost_usd=0.001,
    )


def final(answer: str, reasoning: str = "looked it up") -> LLMResponse:
    return LLMResponse(
        content=f"<REASONING>\n{reasoning}\n</REASONING>\n<FINAL_ANSWER>\n{answer}\n</FINAL_ANSWER>",
        usage=Usage(prompt_tokens=100, completion_tokens=20),
        cost_usd=0.002,
    )


def plain(text: str) -> LLMResponse:
    return LLMResponse(content=text, usage=Usage(prompt_tokens=50, completion_tokens=5), cost_usd=0.0005)


class ScriptedClient:
    """Replays ``turns`` in order (repeating the last one); records every request."""

    def __init__(self, turns: list[Turn]) -> None:
        self.turns = list(turns)
        self.requests: list[list[dict[str, Any]]] = []
        self.tools_seen: list[list[dict[str, Any]]] = []

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        self.requests.append([dict(m) for m in messages])
        self.tools_seen.append(tools)
        idx = min(len(self.requests) - 1, len(self.turns) - 1)
        turn = self.turns[idx]
        return turn(messages) if callable(turn) else turn
