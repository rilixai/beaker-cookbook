"""Ground truth never reaches the agent or its serialized trace."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path

import pytest

from officeqa.agent.agent import OfficeQAAgent
from officeqa.agent.tools import build_toolset
from officeqa.config import RunConfig
from officeqa.data.corpus import Workspace
from officeqa.data.dataset import AgentInput, EvalRecord
from officeqa.data.manifest import build_manifest
from officeqa.runner import run_one_async
from tests.conftest import ScriptedClient, final, tool_call


def test_agent_input_has_exactly_three_fields(records: list[EvalRecord], corpus: Path) -> None:
    assert [f.name for f in dataclasses.fields(AgentInput)] == ["uid", "question", "corpus"]
    inp = records[0].agent_input(build_manifest(corpus))
    assert set(inp.to_json()) == {"uid", "question", "corpus"}
    with pytest.raises((AttributeError, TypeError)):
        inp.answer = "x"  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        AgentInput(uid="u", question="q", corpus=inp.corpus, answer="a")  # type: ignore[call-arg]


def test_eval_record_keeps_ground_truth_separately(records: list[EvalRecord]) -> None:
    r = records[1]
    assert r.answer == "[Massachusetts, 0.866]"
    assert r.source_basenames == ("treasury_bulletin_2013_03", "treasury_bulletin_2013_06")
    assert r.n_source_files == 2 and r.earliest_year == 2013
    assert records[2].n_source_files == 3 and records[2].earliest_year == 1941


def _leaks(serialized: str, record: EvalRecord) -> list[str]:
    found = [record.answer] if record.answer in serialized else []
    found += [b for b in record.source_basenames if b in serialized]
    return found


def test_serialized_trajectory_contains_no_ground_truth(
    workspace: Workspace, records: list[EvalRecord], corpus: Path
) -> None:
    """The scripted agent never touches the source doc, so neither the answer nor any
    source basename may appear anywhere in the serialized trace — including the
    initial user message, the manifest, and tool outputs."""
    record = records[2]  # 3 source files, currency answer
    client = ScriptedClient(
        [
            tool_call("fs_search", path="../officeqa_corpus/parsed", glob="*2013_06*"),
            tool_call("fs_read", path="../officeqa_corpus/parsed/treasury_bulletin_2013_06.txt", head=3),
            final("$1.00"),
        ]
    )
    ts = build_toolset(workspace, ("fs",))
    try:
        ep = asyncio.run(OfficeQAAgent(client, ts).forward(record.agent_input(build_manifest(corpus))))
    finally:
        ts.close()
    serialized = json.dumps(ep.to_json())
    assert _leaks(serialized, record) == []
    for req in client.requests:
        assert _leaks(json.dumps(req), record) == []
    first_user = json.loads(ep.trajectory[1]["content"])
    assert set(first_user) == {"uid", "question", "corpus"}


def test_run_result_json_contains_no_ground_truth(records: list[EvalRecord], corpus: Path, tmp_path: Path) -> None:
    record = records[2]
    cfg = RunConfig(tools=("fs",), corpus="parsed", max_steps=3)
    res = asyncio.run(
        run_one_async(
            record,
            cfg=cfg,
            corpus_root=corpus,
            work_dir=tmp_path / "w",
            client_factory=lambda c: ScriptedClient([final("$2")]),
            isolate=False,
        )
    )
    assert res.correct == 0.0
    assert _leaks(json.dumps(res.to_json()), record) == []


def test_manifest_is_neutral(corpus: Path) -> None:
    m = json.dumps(build_manifest(corpus).to_json())
    assert "treasury_bulletin_" not in m  # lists representations, never individual documents
