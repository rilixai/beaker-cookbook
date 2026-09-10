"""Loop invariants: step cap, sliding window, output truncation, retries, timeouts, resume."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from officeqa.agent.agent import (
    Episode,
    LLMResponse,
    OfficeQAAgent,
    extract_final_answer,
    truncate_output,
    window_messages,
)
from officeqa.agent.tools import build_toolset
from officeqa.config import RunConfig
from officeqa.data.corpus import Workspace, create_workspace
from officeqa.data.dataset import EvalRecord
from officeqa.data.manifest import build_manifest
from officeqa.evaluation.run_eval import select_to_run, summarize
from officeqa.runner import (
    RetryBudget,
    RunResult,
    plurality_vote,
    read_results,
    run_one_async,
    run_split_async,
    write_result,
)
from tests.conftest import ScriptedClient, final, plain, tool_call


def run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


def _agent(workspace: Workspace, client: ScriptedClient, **kw: Any) -> OfficeQAAgent:
    return OfficeQAAgent(client, build_toolset(workspace, ("fs",)), **kw)


def _input(records: list[EvalRecord], corpus: Path, i: int = 0):  # type: ignore[no-untyped-def]
    return records[i].agent_input(build_manifest(corpus))


# --- extraction ---------------------------------------------------------------


def test_extract_final_answer() -> None:
    assert extract_final_answer("<FINAL_ANSWER>\n 42 \n</FINAL_ANSWER>") == "42"
    assert extract_final_answer("<FINAL_ANSWER>1</FINAL_ANSWER> ... <FINAL_ANSWER>2</FINAL_ANSWER>") == "2"
    assert extract_final_answer("<final_answer>x</final_answer>") == "x"
    assert extract_final_answer("no tags here") is None
    assert extract_final_answer(None) is None
    assert extract_final_answer("<FINAL_ANSWER>unclosed") is None


# --- step cap -----------------------------------------------------------------


def test_step_cap_is_enforced(workspace: Workspace, records: list[EvalRecord], corpus: Path) -> None:
    client = ScriptedClient([tool_call("fs_search", path="../officeqa_corpus/parsed")])  # never answers
    agent = _agent(workspace, client, max_steps=7)
    ep = run(agent.forward(_input(records, corpus)))
    assert ep.status == "no_answer" and ep.final_answer is None
    assert ep.n_steps == 7 and len(client.requests) == 7
    assert ep.tool_calls == 7
    # The last-step reminder tells the model to stop calling tools.
    last_reminder = client.requests[-1][-1]["content"]
    assert "last step" in last_reminder
    assert "7 steps remaining" in client.requests[0][-1]["content"]


def test_answer_ends_episode(workspace: Workspace, records: list[EvalRecord], corpus: Path) -> None:
    client = ScriptedClient(
        [tool_call("fs_read", path="../officeqa_corpus/parsed/treasury_bulletin_2013_03.txt"), final("0.866")]
    )
    ep = run(_agent(workspace, client).forward(_input(records, corpus)))
    assert ep.status == "answered" and ep.final_answer == "0.866"
    assert ep.n_steps == 2 and ep.tool_call_counts == {"fs_read": 1}
    assert ep.usage.prompt_tokens == 200 and ep.cost_usd == pytest.approx(0.003)


def test_missing_tag_gets_nudged_then_answers(workspace: Workspace, records: list[EvalRecord], corpus: Path) -> None:
    client = ScriptedClient([plain("The answer is 0.866"), final("0.866")])
    ep = run(_agent(workspace, client).forward(_input(records, corpus)))
    assert ep.status == "answered" and ep.n_steps == 2
    assert any("No <FINAL_ANSWER> tag" in m["content"] for m in client.requests[1] if m["role"] == "user")


# --- sliding window -----------------------------------------------------------


def test_window_messages_pins_system_and_question_and_never_orphans_tool_results() -> None:
    hist: list[dict[str, Any]] = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}]
    for i in range(20):
        hist.append({"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}"}]})
        hist.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"r{i}"})
    win = window_messages(hist, 5)
    assert win[0]["content"] == "S" and win[1]["content"] == "Q"
    assert win[2]["role"] != "tool"  # cut aligned to an assistant turn
    assert len(win) <= 2 + 5
    assert win[-1]["content"] == "r19"
    assert window_messages(hist[:4], 30) == hist[:4]


def test_agent_sends_at_most_window_size_messages(
    workspace: Workspace, records: list[EvalRecord], corpus: Path
) -> None:
    client = ScriptedClient([tool_call("fs_search", path=".")])
    agent = _agent(workspace, client, max_steps=40, window_size=30)
    ep = run(agent.forward(_input(records, corpus)))
    assert len(ep.trajectory) == 2 + 2 * 40  # full trajectory is kept for the trace
    for req in client.requests:
        # system + question + <=30 window + step reminder
        assert len(req) <= 2 + 30 + 1
        assert req[0]["role"] == "system" and req[1]["role"] == "user"
        assert json.loads(req[1]["content"])["uid"] == records[0].uid
    assert len(client.requests[-1]) == 2 + 30 + 1


# --- truncation ---------------------------------------------------------------


def test_truncate_output_caps_at_limit() -> None:
    text, cut = truncate_output("x" * 100, 1000)
    assert not cut and text == "x" * 100
    big = "".join(f"line{i}\n" for i in range(50_000))
    text, cut = truncate_output(big, 25_000)
    assert cut and len(text) <= 25_000
    assert "truncated" in text and text.startswith("line0\n") and text.endswith("line49999\n")


def test_tool_outputs_are_truncated_in_trajectory(
    workspace: Workspace, records: list[EvalRecord], corpus: Path
) -> None:
    client = ScriptedClient([tool_call("python_exec", code="print('z' * 100000)"), final("1")])
    ts = build_toolset(workspace, ("repl",))
    try:
        agent = OfficeQAAgent(client, ts, tool_output_limit=25_000)
        ep = run(agent.forward(_input(records, corpus)))
    finally:
        ts.close()
    tool_msgs = [m for m in ep.trajectory if m["role"] == "tool"]
    assert len(tool_msgs) == 1 and len(tool_msgs[0]["content"]) <= 25_000
    assert ep.truncated_tool_outputs == 1
    assert len(client.requests[1][-2]["content"]) <= 25_000


# --- retries / resume ---------------------------------------------------------


class FlakyFactory:
    """Raises on the first ``fail`` client constructions, then returns a scripted client."""

    def __init__(self, fail: int, turns: list[LLMResponse]) -> None:
        self.fail = fail
        self.turns = turns
        self.calls = 0

    def __call__(self, cfg: RunConfig) -> ScriptedClient:
        self.calls += 1
        if self.calls <= self.fail:
            raise ConnectionError("provider down")
        return ScriptedClient(self.turns)


def _cfg(**kw: Any) -> RunConfig:
    base: dict[str, Any] = {"tools": ("fs",), "corpus": "parsed", "max_steps": 5, "max_retries": 3}
    base.update(kw)
    return RunConfig(**base)


def test_crash_is_retried_from_budget(records: list[EvalRecord], corpus: Path, tmp_path: Path) -> None:
    factory = FlakyFactory(fail=2, turns=[final("9876543.21")])
    budget = RetryBudget(3)
    res = run(
        run_one_async(
            records[0],
            cfg=_cfg(),
            corpus_root=corpus,
            work_dir=tmp_path / "w",
            client_factory=factory,
            retry_budget=budget,
            isolate=False,
        )
    )
    assert res.status == "answered" and res.attempts == 3 and budget.used == 2
    assert res.correct == 1.0 and res.scores == {"0.0": 1.0, "0.001": 1.0, "0.01": 1.0, "0.05": 1.0}


def test_exhausted_budget_yields_error_scored_zero(records: list[EvalRecord], corpus: Path, tmp_path: Path) -> None:
    factory = FlakyFactory(fail=99, turns=[final("1")])
    res = run(
        run_one_async(
            records[0],
            cfg=_cfg(),
            corpus_root=corpus,
            work_dir=tmp_path / "w",
            client_factory=factory,
            retry_budget=RetryBudget(2),
            isolate=False,
        )
    )
    assert res.status == "error" and res.attempts == 3 and res.correct == 0.0
    assert res.error is not None and "retry budget exhausted" in res.error
    assert not res.clean


def test_timeout_scores_zero_keeps_partial_trajectory(records: list[EvalRecord], corpus: Path, tmp_path: Path) -> None:
    class SlowClient:
        async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
            await asyncio.sleep(5)
            return final("x")

    res = run(
        run_one_async(
            records[0],
            cfg=_cfg(task_timeout_s=0.3),
            corpus_root=corpus,
            work_dir=tmp_path / "w",
            client_factory=lambda cfg: SlowClient(),
            isolate=False,
        )
    )
    assert res.status == "timeout" and res.final_answer is None and res.correct == 0.0
    assert res.trajectory[0]["role"] == "system" and res.trajectory[1]["role"] == "user"
    assert res.error is not None and "timed out" in res.error and not res.clean


def test_resume_reuses_clean_reruns_unclean(records: list[EvalRecord], corpus: Path, tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    cfg = _cfg()

    def factory(_cfg: RunConfig) -> ScriptedClient:
        return ScriptedClient([final("9876543.21")])

    first = run(
        run_one_async(
            records[0], cfg=cfg, corpus_root=corpus, work_dir=tmp_path / "w", client_factory=factory, isolate=False
        )
    )
    write_result(first, results_dir)
    broken = RunResult.from_json(
        first.to_json() | {"uid": records[1].uid, "status": "timeout", "final_answer": None, "scores": {"0.0": 0.0}}
    )
    write_result(broken, results_dir)

    existing = read_results(results_dir)
    assert set(existing) == {records[0].uid, records[1].uid}
    todo = select_to_run(records, existing)
    assert [r.uid for r in todo] == [records[1].uid, records[2].uid]  # clean q1 reused; timeout + missing re-run
    assert [r.uid for r in select_to_run(records, existing, rerun=True)] == [r.uid for r in records]

    # Timeouts/errors are included (as 0) in aggregates, never dropped.
    s = summarize(list(existing.values()), expected=[r.uid for r in records])
    assert s["n"] == 3 and s["n_scored"] == 2 and s["missing"] == [records[2].uid]
    assert s["correctness"]["0%"] == pytest.approx(1 / 3)
    assert s["statuses"] == {"answered": 1, "timeout": 1, "missing": 1}


def test_run_split_concurrency_and_queue_wait(records: list[EvalRecord], corpus: Path, tmp_path: Path) -> None:
    seen: list[str] = []

    def factory(_cfg: RunConfig) -> ScriptedClient:
        return ScriptedClient([final("nope")])

    res = run(
        run_split_async(
            records,
            _cfg(max_concurrent=1),
            corpus_root=corpus,
            work_dir=tmp_path / "w",
            client_factory=factory,
            on_result=lambda r: seen.append(r.uid),
            isolate=False,
        )
    )
    assert sorted(seen) == sorted(r.uid for r in records)
    assert all(r.status == "answered" and r.correct == 0.0 for r in res)
    assert all(r.latency_s > 0 and r.queue_wait_s >= 0 for r in res)
    assert sum(1 for r in res if r.queue_wait_s > 0) >= 1  # with concurrency 1 some sample had to wait


def test_plurality_vote_and_multi_rollout(records: list[EvalRecord], corpus: Path, tmp_path: Path) -> None:
    assert plurality_vote(["1", "2", "2"], seed="s") == 1
    assert plurality_vote(["$1,000", "1000", "5"], seed="s") in (0, 1)
    assert plurality_vote([None, None], seed="s") == 0
    answers = iter(["1", "9876543.21", "9876543.21"])

    def factory(_cfg: RunConfig) -> ScriptedClient:
        return ScriptedClient([final(next(answers))])

    res = run(
        run_one_async(
            records[0],
            cfg=_cfg(n_rollouts=3),
            corpus_root=corpus,
            work_dir=tmp_path / "w",
            client_factory=factory,
            isolate=False,
        )
    )
    assert res.final_answer == "9876543.21" and res.correct == 1.0
    assert len(res.rollouts) == 3 and sum(r["chosen"] for r in res.rollouts) == 1
    assert res.steps == 3 and res.attempts == 3


def test_run_result_roundtrip(records: list[EvalRecord], corpus: Path, tmp_path: Path) -> None:
    res = run(
        run_one_async(
            records[2],
            cfg=_cfg(),
            corpus_root=corpus,
            work_dir=tmp_path / "w",
            client_factory=lambda c: ScriptedClient([final("$7,046,001.98")]),
            isolate=False,
        )
    )
    path = write_result(res, tmp_path / "results")
    back = read_results(tmp_path / "results")[records[2].uid]
    assert back == res and json.loads(path.read_text())["correct"] == 1.0


def test_workspace_layout(tmp_path: Path, corpus: Path) -> None:
    ws = create_workspace(tmp_path / "w", "u1", corpus, isolate=False)
    assert ws.cwd == tmp_path / "w" / "u1" / "cwd" and ws.cwd.is_dir()
    assert (ws.cwd / ".." / "officeqa_corpus" / "pdfs").resolve().is_dir()
    assert (ws.cwd / ".." / "officeqa_corpus" / "parsed" / "treasury_bulletin_2013_03.txt").resolve().is_file()
    (ws.cwd / "junk").write_text("x")
    ws2 = create_workspace(tmp_path / "w", "u1", corpus, isolate=False, fresh=True)
    assert not (ws2.cwd / "junk").exists()


def test_search_arms_other_than_fs_are_stubbed(records: list[EvalRecord], corpus: Path, tmp_path: Path) -> None:
    res = run(
        run_one_async(
            records[0],
            cfg=_cfg(search="vs", max_retries=0),
            corpus_root=corpus,
            work_dir=tmp_path / "w",
            client_factory=lambda c: ScriptedClient([final("1")]),
            isolate=False,
        )
    )
    assert res.status == "error" and res.error is not None and "stubbed" in res.error


def test_episode_json_is_serializable() -> None:
    ep = Episode()
    ep.trajectory.append({"role": "system", "content": "s"})
    json.dumps(ep.to_json())
