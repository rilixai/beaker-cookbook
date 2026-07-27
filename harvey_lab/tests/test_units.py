"""Hermetic unit + integration tests for the Harvey LAB agent.

Zero network: the agent runs on a scripted Stirrup client over the *local*
code-execution backend (a temp directory — real shell, no network), the rubric
judge is a stub, and task documents come from a fixture tree on disk. Covers
the data loader (incl. nested task discovery), the frozen splits, the
workspace staging + deliverable text extraction, the batched verdict parser,
LAB-AA's exact-filename / partial-submission scoring rules, the finish and
abandon tools, and one full agent -> judge evaluation pass.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from pathlib import Path
from typing import Any

import pytest

from harvey_lab import cli as cli_mod
from harvey_lab.agent.agent import (
    FinishParams,
    HarveyLabAgent,
    HarveyLabAgentOutput,
    _build_finish_tools,
    _count_turns,
    _render_template,
)
from harvey_lab.agent.workspace import TaskWorkspace, extract_text, task_source_from_dir
from harvey_lab.config import HarveyLabConfig
from harvey_lab.data import fetch as fetch_mod
from harvey_lab.data.dataset import load_records, read_split
from harvey_lab.evaluation.run_eval import evaluate_agent_on_records, evaluate_outputs_on_records, evaluate_record
from harvey_lab.evaluation.scoring import (
    ALL_PASS_FIELD,
    CRITERION_PASS_RATE_FIELD,
    JudgeCallError,
    _parse_batch_verdicts,
    _scope_deliverables,
    build_rubric_judge,
    score_rubric,
)


# ─── fixtures ─────────────────────────────────────────────────────────


def _write_task(root: Path, task_id: str, *, criteria: list[dict], deliverables: dict[str, str]) -> None:
    task_dir = root / task_id
    (task_dir / "documents").mkdir(parents=True, exist_ok=True)
    (task_dir / "documents" / "notes.txt").write_text(
        "Master Services Agreement between Acme and Beta. Termination fee: $50,000.",
        encoding="utf-8",
    )
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "title": f"task {task_id}",
                "work_type": "analyze",
                "tags": [task_id.split("/", 1)[0]],
                "instructions": "Summarize the termination fee and cite the source document.",
                "deliverables": deliverables,
                "criteria": criteria,
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def tasks_root(tmp_path: Path) -> Path:
    root = tmp_path / "tasks"
    crit = [
        {"id": "C1", "title": "States the fee", "match_criteria": "Mentions $50,000.", "deliverables": ["memo.md"]},
        {"id": "C2", "title": "Cites source", "match_criteria": "Cites notes.txt.", "deliverables": ["memo.md"]},
    ]
    _write_task(root, "contracts/t1", criteria=crit, deliverables={"memo.md": "Memo"})
    _write_task(root, "contracts/t2", criteria=crit, deliverables={"memo.md": "Memo"})
    _write_task(root, "corporate-ma/t1", criteria=crit, deliverables={"memo.md": "Memo"})
    _write_task(root, "tax/t1", criteria=crit, deliverables={"memo.md": "Memo"})
    _write_task(root, "litigation/t1", criteria=crit, deliverables={"memo.md": "Memo"})
    # A nested task (larger areas nest sub-categories under the practice area).
    _write_task(root, "contracts/banking/deep", criteria=crit, deliverables={"memo.md": "Memo"})
    return root


# ─── scripted Stirrup client ──────────────────────────────────────────


class _ScriptedClient:
    """A Stirrup ``LLMClient`` that replays a fixed tool-call script.

    Turn 1 writes the deliverable through ``code_exec`` (as the real agent
    must — there is no ``write_deliverable`` tool); turn 2 submits it via
    ``finish``. Never touches the network.
    """

    def __init__(self, *, deliverable_body: str, deliverable_name: str = "memo.md") -> None:
        self._deliverable_body = deliverable_body
        self._deliverable_name = deliverable_name
        self._turn = 0

    @property
    def model_slug(self) -> str:
        return "scripted/test"

    @property
    def max_tokens(self) -> int:
        return 100_000

    async def generate(self, messages: list[Any], tools: dict[str, Any]) -> Any:
        from stirrup.core.models import AssistantMessage, ToolCall

        workspace = _workspace_from_messages(messages)
        self._turn += 1
        if self._turn == 1:
            heredoc = f"cat > {self._deliverable_name} <<'HLEOF'\n{self._deliverable_body}\nHLEOF"
            call = ToolCall(
                name="code_exec",
                arguments=json.dumps({"cmd": heredoc}),
                tool_call_id="tc-1",
            )
        else:
            call = ToolCall(
                name="finish",
                arguments=json.dumps(
                    {"summary": "Deliverable written.", "paths": [f"{workspace}/{self._deliverable_name}"]},
                ),
                tool_call_id="tc-2",
            )
        return AssistantMessage(content="", tool_calls=[call])


def _workspace_from_messages(messages: list[Any]) -> str:
    for message in messages:
        content = str(getattr(message, "content", ""))
        match = re.search(r"working directory, `([^`]+)`", content)
        if match:
            return match.group(1)
    raise AssertionError("Task prompt did not include the execution working directory.")


def _scripted_model_factory(body: str, name: str = "memo.md") -> Any:
    def _factory(_model: str, _temp: float, _max_tokens: int, _timeout: float) -> Any:
        return _ScriptedClient(deliverable_body=body, deliverable_name=name)

    return _factory


def _local_exec_factory(config: Any) -> Any:
    """A local (temp-dir) code-exec backend with a short per-command timeout."""
    from stirrup.tools.code_backends.local import LocalCodeExecToolProvider

    return LocalCodeExecToolProvider(shell_timeout=30)


def _build_agent(tasks_root: Path, body: str, name: str = "memo.md", **cfg: Any) -> HarveyLabAgent:
    return HarveyLabAgent(
        config=HarveyLabConfig(max_turns=5, enable_view_image=False, **cfg),
        task_source=task_source_from_dir(tasks_root),
        model_factory=_scripted_model_factory(body, name),
        exec_provider_factory=_local_exec_factory,
    )


def _fee_judge(_desc: str, criteria: list[dict], out: str) -> dict[str, bool]:
    """Batched stub judge: a criterion passes if the fee/source appears."""
    passed = "$50,000" in out or "notes.txt" in out
    return {str(c["id"]): passed for c in criteria}


def _all_pass_judge(_desc: str, criteria: list[dict], _out: str) -> dict[str, bool]:
    return {str(c["id"]): True for c in criteria}


def _reject_judge(_desc: str, criteria: list[dict], _out: str) -> dict[str, bool]:
    return {str(c["id"]): False for c in criteria}


# ─── data loader ──────────────────────────────────────────────────────


def test_load_records(tasks_root: Path) -> None:
    records = load_records(tasks_root)
    assert len(records) == 6
    rec = next(r for r in records if r.task_id == "contracts/t1")
    assert rec.practice_area == "contracts"
    assert rec.deliverable_names == ("memo.md",)
    assert rec.documents == ("notes.txt",)
    assert len(rec.criteria) == 2


def test_load_records_discovers_nested_tasks(tasks_root: Path) -> None:
    """Larger areas nest sub-categories; discovery must walk recursively."""
    records = load_records(tasks_root)
    nested = next(r for r in records if r.task_id == "contracts/banking/deep")
    assert nested.practice_area == "contracts"  # first path segment


def test_load_records_by_task_ids_preserves_order(tasks_root: Path) -> None:
    ids = ["tax/t1", "contracts/t1"]
    records = load_records(tasks_root, task_ids=ids)
    assert [r.task_id for r in records] == ids


def test_task_fingerprint_tracks_metadata_and_document_content(tasks_root: Path) -> None:
    task_id = "contracts/t1"
    first = load_records(tasks_root, task_ids=[task_id])[0].task_fingerprint

    notes = tasks_root / task_id / "documents/notes.txt"
    notes.write_text(notes.read_text() + "changed")
    second = load_records(tasks_root, task_ids=[task_id])[0].task_fingerprint

    task_path = tasks_root / task_id / "task.json"
    task = json.loads(task_path.read_text())
    task["instructions"] += " changed"
    task_path.write_text(json.dumps(task))
    third = load_records(tasks_root, task_ids=[task_id])[0].task_fingerprint

    assert len({first, second, third}) == 3


def test_load_records_raises_on_missing_split_task(tasks_root: Path) -> None:
    # A frozen-split id absent from the tree must fail loudly (checkout drift),
    # not silently shrink the run.
    with pytest.raises(FileNotFoundError, match="no-such/task"):
        load_records(tasks_root, task_ids=["contracts/t1", "no-such/task"])


# ─── on-demand task fetch ─────────────────────────────────────────────


def _fake_github(tree: dict[str, bytes]) -> Any:
    """Return a fake ``_request`` serving the contents API, Trees API + raw blobs.

    ``tree`` maps repo-relative file paths (e.g. ``tasks/x/task.json``) to bytes.
    Directory listings and the recursive tree are synthesized from the paths; a
    dir's synthetic ``sha`` is its own path so the Trees API can resolve it.
    """
    import urllib.parse

    def _request(url: str) -> bytes:
        if url.startswith("https://raw.githubusercontent.com/"):
            # .../<owner>/<name>/<ref>/<path...>
            _owner, _name, _ref, *parts = url[len("https://raw.githubusercontent.com/") :].split("/")
            return tree[urllib.parse.unquote("/".join(parts))]
        if "/git/trees/" in url:
            root = urllib.parse.unquote(url.split("/git/trees/", 1)[1].split("?", 1)[0])
            prefix = root + "/"
            items: list[dict[str, str]] = []
            seen: set[str] = set()
            for fpath in tree:
                if not fpath.startswith(prefix):
                    continue
                rel = fpath[len(prefix) :]
                parts = rel.split("/")
                for i in range(len(parts) - 1):
                    d = "/".join(parts[: i + 1])
                    if d not in seen:
                        seen.add(d)
                        items.append({"path": d, "type": "tree"})
                items.append({"path": rel, "type": "blob", "size": len(tree[fpath])})
            return json.dumps({"tree": items, "truncated": False}).encode()
        # contents API: ".../contents/<path>?ref=..." (path is percent-encoded)
        path = urllib.parse.unquote(url.split("/contents/", 1)[1].split("?", 1)[0])
        prefix = path + "/"
        children: dict[str, str] = {}
        for fpath in tree:
            if fpath.startswith(prefix):
                head = fpath[len(prefix) :].split("/", 1)
                children[head[0]] = "dir" if len(head) > 1 else "file"
        entries = []
        for name, kind in sorted(children.items()):
            entry = {"name": name, "type": kind, "sha": f"{path}/{name}"}
            if kind == "file":
                entry["size"] = len(tree[f"{path}/{name}"])
            entries.append(entry)
        return json.dumps(entries).encode()

    return _request


def test_ensure_task_dirs_fetches_and_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tree = {
        "tasks/contracts/t1/task.json": b'{"title": "T1"}',
        "tasks/contracts/t1/documents/notes.txt": b"hello",
        "tasks/contracts/t1/documents/sub/deep.txt": b"deep",
        # LAB folders can contain spaces; the contents-API path must be encoded.
        "tasks/contracts/t1/documents/1.0 Transaction Documents/spa ced.txt": b"x",
    }
    calls = {"n": 0}

    def _counting(url: str) -> bytes:
        calls["n"] += 1
        return _fake_github(tree)(url)

    monkeypatch.setattr(fetch_mod, "_request", _counting)
    tasks_root = fetch_mod.ensure_task_dirs(["contracts/t1"], cache_dir=tmp_path)

    assert (tasks_root / "contracts/t1/task.json").read_bytes() == b'{"title": "T1"}'
    assert (tasks_root / "contracts/t1/documents/sub/deep.txt").read_bytes() == b"deep"
    assert (tasks_root / "contracts/t1/documents/1.0 Transaction Documents/spa ced.txt").read_bytes() == b"x"
    # The fetched tree loads through the normal record loader + workspace factory.
    records = load_records(tasks_root, task_ids=["contracts/t1"])
    assert records[0].documents == (
        "1.0 Transaction Documents/spa ced.txt",
        "notes.txt",
        "sub/deep.txt",
    )

    # A second call hits the sentinel and makes zero network requests.
    calls["n"] = 0
    fetch_mod.ensure_task_dirs(["contracts/t1"], cache_dir=tmp_path)
    assert calls["n"] == 0


def test_ensure_task_dirs_resumes_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # An interrupted task (no `.fetched` sentinel, but a `.partial` marker for
    # this commit + some files already on disk) resumes: correctly-sized files
    # are not re-downloaded, only the missing ones are.
    tree = {
        "tasks/contracts/t1/task.json": b'{"title": "T1"}',
        "tasks/contracts/t1/documents/a.txt": b"aaa",
        "tasks/contracts/t1/documents/b.txt": b"bbbb",
    }
    # Simulate a prior interrupted run: task.json already fetched, plus the marker.
    commit = fetch_mod.HARVEY_LABS_COMMIT
    (tmp_path / "tasks/contracts/t1").mkdir(parents=True)
    (tmp_path / "tasks/contracts/t1/task.json").write_bytes(b'{"title": "T1"}')
    (tmp_path / ".partial/contracts").mkdir(parents=True)
    (tmp_path / ".partial/contracts/t1").write_text(commit)

    fetched: list[str] = []
    base = _fake_github(tree)

    def _record(url: str) -> bytes:
        if url.startswith("https://raw.githubusercontent.com/"):
            fetched.append(url.rsplit("/", 1)[1])
        return base(url)

    monkeypatch.setattr(fetch_mod, "_request", _record)
    root = fetch_mod.ensure_task_dirs(["contracts/t1"], cache_dir=tmp_path)

    # task.json was already on disk at the right size -> not re-downloaded.
    assert "task.json" not in fetched
    assert sorted(fetched) == ["a.txt", "b.txt"]
    assert (root / "contracts/t1/documents/b.txt").read_bytes() == b"bbbb"
    # The partial marker is cleared once the task completes.
    assert not (tmp_path / ".partial/contracts/t1").exists()


def test_ensure_task_dirs_refetches_on_commit_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A reused --cache-dir must not serve a stale tree after a pin bump: the
    # sentinel records the commit, and a mismatch forces a refetch.
    old_tree = {"tasks/contracts/t1/task.json": b'{"title": "old"}'}
    monkeypatch.setattr(fetch_mod, "_request", _fake_github(old_tree))
    fetch_mod.ensure_task_dirs(["contracts/t1"], commit="aaa", cache_dir=tmp_path)

    new_tree = {"tasks/contracts/t1/task.json": b'{"title": "new"}'}
    calls = {"n": 0}

    def _counting(url: str) -> bytes:
        calls["n"] += 1
        return _fake_github(new_tree)(url)

    monkeypatch.setattr(fetch_mod, "_request", _counting)
    root = fetch_mod.ensure_task_dirs(["contracts/t1"], commit="bbb", cache_dir=tmp_path)
    assert calls["n"] > 0
    assert (root / "contracts/t1/task.json").read_bytes() == b'{"title": "new"}'


# ─── frozen splits ────────────────────────────────────────────────────


def test_frozen_splits_are_disjoint_and_capped() -> None:
    train, val, test = read_split("train"), read_split("val"), read_split("test")
    assert len(val) == 100
    assert len(test) == 100
    assert len(train) > 0
    # Three-way disjoint, and every id is a practice-area-prefixed path.
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    assert all("/" in tid for tid in val)


# ─── workspace ────────────────────────────────────────────────────────


def test_workspace_stages_documents_and_collects_named_deliverables(tmp_path: Path) -> None:
    ws = TaskWorkspace(tmp_path / "ws")
    (ws.documents_dir / "a.txt").write_text("hello WORLD line", encoding="utf-8")

    # Deliverables are pulled back into output/ by the agent, then collected
    # by EXACT requested filename — anything else the agent left behind is not
    # graded.
    ws.deliverable_path("memo.md").write_text("final", encoding="utf-8")
    ws.deliverable_path("scratch.md").write_text("ignore me", encoding="utf-8")
    assert ws.collect_deliverables(["memo.md"]) == {"memo.md": "final"}
    # A requested file the agent never produced is simply absent.
    assert ws.collect_deliverables(["memo.md", "missing.docx"]) == {"memo.md": "final"}


def test_workspace_rejects_escape(tmp_path: Path) -> None:
    ws = TaskWorkspace(tmp_path / "ws")
    with pytest.raises(ValueError):
        ws.deliverable_path("../escape.txt")


def test_extract_text_falls_back_and_survives_bad_binaries(tmp_path: Path) -> None:
    plain = tmp_path / "note.md"
    plain.write_text("# Memo\nbody", encoding="utf-8")
    assert "# Memo" in extract_text(plain)
    assert extract_text(plain, max_chars=3) == "# M"

    # A file claiming to be .docx but containing garbage must yield a note, not
    # raise — one unreadable deliverable cannot abort a grading run.
    broken = tmp_path / "broken.docx"
    broken.write_bytes(b"not a docx at all")
    assert "broken.docx" in extract_text(broken)


# ─── batched verdict parsing + scoring ────────────────────────────────


def test_parse_batch_verdicts_maps_ids() -> None:
    reply = '{"verdicts": [{"id": "C1", "verdict": "pass"}, {"id": "C2", "verdict": "fail"}]}'
    assert _parse_batch_verdicts(reply, ["C1", "C2"]) == {"C1": True, "C2": False}


def test_parse_batch_verdicts_missing_ids_default_fail() -> None:
    # C2 omitted by the judge -> conservative FAIL (must not inflate).
    reply = 'noise {"verdicts": [{"id": "C1", "verdict": "pass"}]} trailing'
    assert _parse_batch_verdicts(reply, ["C1", "C2"]) == {"C1": True, "C2": False}


def test_parse_batch_verdicts_garbage_is_judge_error() -> None:
    with pytest.raises(JudgeCallError, match="no usable verdicts"):
        _parse_batch_verdicts("no json here", ["C1"])


def test_parse_batch_verdicts_null_verdicts_is_judge_error() -> None:
    with pytest.raises(JudgeCallError, match="no usable verdicts"):
        _parse_batch_verdicts('{"verdicts": null}', ["C1"])


def test_parse_batch_verdicts_tolerates_surrounding_prose_and_fences() -> None:
    # A code fence around the JSON, plus trailing prose that itself has braces:
    # the balanced-brace scan must recover the verdicts object, not over-capture.
    reply = (
        "Here is my grading:\n"
        '```json\n{"verdicts": [{"id": "C1", "verdict": "pass"}, {"id": "C2", "verdict": "pass"}]}\n```\n'
        "Note: formatting like {curly} should not break parsing."
    )
    assert _parse_batch_verdicts(reply, ["C1", "C2"]) == {"C1": True, "C2": True}


def test_build_rubric_judge_uses_injected_llm() -> None:
    """The judge sends all batched criteria in one call and parses the reply."""
    calls: list[list[dict]] = []

    def _llm(*, model: str, messages: list[dict]) -> str:
        calls.append(messages)
        return '{"verdicts": [{"id": "C1", "verdict": "pass"}, {"id": "C2", "verdict": "fail"}]}'

    judge = build_rubric_judge(model="stub/test", llm=_llm)
    criteria = [{"id": "C1", "title": "t", "match_criteria": "x"}, {"id": "C2", "title": "t", "match_criteria": "y"}]
    assert judge("task", criteria, "output") == {"C1": True, "C2": False}
    assert len(calls) == 1  # both criteria graded in a single batched call


def test_build_rubric_judge_propagates_provider_errors() -> None:
    def _broken_llm(**_kwargs: Any) -> str:
        raise RuntimeError("context window exceeded")

    judge = build_rubric_judge(model="stub/test", llm=_broken_llm)
    with pytest.raises(RuntimeError, match="context window exceeded"):
        judge("task", [{"id": "C1", "match_criteria": "x"}], "output")


def test_scope_deliverables_selects_named_only() -> None:
    scoped = _scope_deliverables(["memo.md"], {"memo.md": "A", "appendix.md": "B"})
    assert "A" in scoped and "B" not in scoped


def test_scope_deliverables_preserves_full_text() -> None:
    content = "A" * 50_000 + "evidence at the end"
    scoped = _scope_deliverables(["memo.md"], {"memo.md": content})
    assert scoped.endswith("evidence at the end")


def test_scope_deliverables_marks_partial_submission_absent() -> None:
    """A criterion spanning two files, only one produced, is still judged — with
    the missing file explicitly marked absent (LAB-AA's partial-submission rule)."""
    scoped = _scope_deliverables(["memo.md", "schedule.xlsx"], {"memo.md": "A"})
    assert "A" in scoped
    assert "schedule.xlsx" in scoped
    assert "not produced" in scoped


def test_scope_deliverables_does_not_fuzzy_match_filenames() -> None:
    """LAB-AA requires EXACT filenames: a near-miss counts as not produced."""
    scoped = _scope_deliverables(["memo.md"], {"my-memo.md": "A"})
    assert "A" not in scoped
    assert "not produced" in scoped


def test_score_rubric_fails_unproduced_scope_without_calling_judge() -> None:
    """When NONE of a criterion's deliverables exist it fails outright, and the
    judge is never invoked for it."""
    criteria = [
        {"id": "C1", "title": "t", "match_criteria": "x", "deliverables": ["memo.md"]},
        {"id": "C2", "title": "t", "match_criteria": "y", "deliverables": ["schedule.xlsx"]},
    ]
    seen: list[str] = []

    def _judge(_desc: str, crits: list[dict], _out: str) -> dict[str, bool]:
        seen.extend(str(c["id"]) for c in crits)
        return {str(c["id"]): True for c in crits}

    result = score_rubric(
        criteria=criteria,
        deliverables={"memo.md": "body"},
        task_description="t",
        judge=_judge,
    )
    assert seen == ["C1"]  # C2's only deliverable is missing -> never judged
    assert result["passed"] == 1
    assert result[CRITERION_PASS_RATE_FIELD] == 0.5
    assert result[ALL_PASS_FIELD] == 0.0


def test_score_rubric_all_and_partial() -> None:
    criteria = [
        {"id": "C1", "title": "t", "match_criteria": "x", "deliverables": ["m.md"]},
        {"id": "C2", "title": "t", "match_criteria": "y", "deliverables": ["m.md"]},
    ]
    deliverables = {"m.md": "body"}
    result_all = score_rubric(
        criteria=criteria, deliverables=deliverables, task_description="t", judge=_all_pass_judge
    )
    assert result_all[ALL_PASS_FIELD] == 1.0
    assert result_all[CRITERION_PASS_RATE_FIELD] == 1.0
    assert result_all["passed"] == 2
    assert result_all["total_criteria"] == 2

    def _half(_desc: str, crits: list[dict], _out: str) -> dict[str, bool]:
        return {str(c["id"]): c["id"] == "C1" for c in crits}

    result_partial = score_rubric(criteria=criteria, deliverables=deliverables, task_description="t", judge=_half)
    assert result_partial[ALL_PASS_FIELD] == 0.0
    assert result_partial[CRITERION_PASS_RATE_FIELD] == 0.5


def test_score_rubric_batches_by_size() -> None:
    """Same-scope criteria are chunked into batches of ``batch_size``."""
    criteria = [{"id": f"C{i}", "title": "t", "match_criteria": "x", "deliverables": ["m.md"]} for i in range(5)]
    batch_sizes: list[int] = []
    progress: list[tuple[int, int, int]] = []

    def _judge(_desc: str, crits: list[dict], _out: str) -> dict[str, bool]:
        batch_sizes.append(len(crits))
        return {str(c["id"]): True for c in crits}

    result = score_rubric(
        criteria=criteria,
        deliverables={"m.md": "b"},
        task_description="t",
        judge=_judge,
        batch_size=2,
        judge_batch_callback=lambda start, end, total: progress.append((start, end, total)),
    )
    assert result[ALL_PASS_FIELD] == 1.0
    assert batch_sizes == [2, 2, 1]  # 5 criteria, one scope, chunks of 2
    assert progress == [(1, 2, 5), (3, 4, 5), (5, 5, 5)]


def test_score_rubric_empty_rubric_is_unscoreable() -> None:
    result = score_rubric(criteria=[], deliverables={}, task_description="t", judge=_all_pass_judge)
    assert result["total_criteria"] == 0


def test_score_rubric_propagates_judge_errors() -> None:
    def _broken_judge(_desc: str, _criteria: list[dict], _output: str) -> dict[str, bool]:
        raise RuntimeError("context window exceeded")

    with pytest.raises(RuntimeError, match="context window exceeded"):
        score_rubric(
            criteria=[{"id": "C1", "match_criteria": "x", "deliverables": ["memo.md"]}],
            deliverables={"memo.md": "body"},
            task_description="t",
            judge=_broken_judge,
        )


def test_render_template_substitutes_and_appends_instructions() -> None:
    rendered = _render_template(
        "{{instructions}}\n\n{{ deliverables }}",
        {"instructions": "do it", "deliverables": "- x"},
    )
    assert "do it" in rendered and "- x" in rendered
    # A template that drops `instructions` still receives them (the agent must
    # always see the task); other unreferenced vars are simply not injected.
    dropped = _render_template("no vars here", {"instructions": "I", "workspace_dir": "/tmp/x"})
    assert "I" in dropped
    assert "/tmp/x" not in dropped


# ─── full agent → judge integration ──────────────────────────────────


def test_evaluate_record_all_pass(tasks_root: Path) -> None:
    agent = _build_agent(tasks_root, "Termination fee is $50,000, per notes.txt.")
    record = load_records(tasks_root, task_ids=["contracts/t1"])[0]
    result = asyncio.run(evaluate_record(record=record, agent=agent, judge=_fee_judge))
    assert result["kind"] == "scored"
    assert result[ALL_PASS_FIELD] == 1.0
    assert result[CRITERION_PASS_RATE_FIELD] == 1.0


def test_evaluate_record_fail_when_judge_rejects(tasks_root: Path) -> None:
    agent = _build_agent(tasks_root, "irrelevant content")
    record = load_records(tasks_root, task_ids=["tax/t1"])[0]
    result = asyncio.run(evaluate_record(record=record, agent=agent, judge=_reject_judge))
    assert result[ALL_PASS_FIELD] == 0.0
    assert result[CRITERION_PASS_RATE_FIELD] == 0.0


class _CapturingClient:
    """Records every message/tool set it sees, then replays ``script`` calls."""

    def __init__(self, script: list[tuple[str, dict[str, Any]]]) -> None:
        self._script = script
        self._turn = 0
        self.messages: list[Any] = []
        self.tool_names: list[str] = []

    @property
    def model_slug(self) -> str:
        return "scripted/test"

    @property
    def max_tokens(self) -> int:
        return 100_000

    async def generate(self, messages: list[Any], tools: dict[str, Any]) -> Any:
        from stirrup.core.models import AssistantMessage, ToolCall

        if not self.messages:
            self.messages = list(messages)
            self.tool_names = sorted(tools)
        name, args = self._script[min(self._turn, len(self._script) - 1)]
        args = dict(args)
        if name == "finish":
            workspace = _workspace_from_messages(messages)
            args["paths"] = [p if Path(p).is_absolute() else f"{workspace}/{p}" for p in args.get("paths", [])]
        self._turn += 1
        return AssistantMessage(
            content="",
            tool_calls=[ToolCall(name=name, arguments=json.dumps(args), tool_call_id=f"tc-{self._turn}")],
        )


def _run_with_client(tasks_root: Path, client: _CapturingClient, task_id: str = "contracts/t1", **kw: Any) -> Any:
    agent = HarveyLabAgent(
        config=HarveyLabConfig(max_turns=4, enable_view_image=False),
        task_source=task_source_from_dir(tasks_root),
        model_factory=lambda *_: client,
        exec_provider_factory=_local_exec_factory,
        **kw,
    )
    record = load_records(tasks_root, task_ids=[task_id])[0]
    return asyncio.run(agent.forward(record=record))


def test_forward_uses_configured_prompts(tasks_root: Path) -> None:
    """The system prompt the LLM sees is the agent's configured prompt."""
    client = _CapturingClient([("abandon_task_finish", {"reason": "done"})])
    _run_with_client(tasks_root, client, system_prompt="MY-CUSTOM-SYSTEM-PROMPT")
    assert "MY-CUSTOM-SYSTEM-PROMPT" in str(getattr(client.messages[0], "content", ""))


def test_forward_exposes_only_code_exec_and_finish_tools(tasks_root: Path) -> None:
    """LAB-AA gives the model a single code-execution tool plus the finish pair —
    no curated read/write/grep helpers."""
    client = _CapturingClient([("abandon_task_finish", {"reason": "nothing to do"})])
    _run_with_client(tasks_root, client)
    assert client.tool_names == ["abandon_task_finish", "code_exec", "finish"]


def test_forward_uploads_documents_into_the_environment(tasks_root: Path) -> None:
    """The task's documents must be readable through `code_exec` at documents/."""
    client = _CapturingClient(
        [
            ("code_exec", {"cmd": "cp documents/notes.txt memo.md"}),
            ("finish", {"summary": "copied", "paths": ["memo.md"]}),
        ]
    )
    output = _run_with_client(tasks_root, client)
    assert "$50,000" in output.deliverables["memo.md"]
    assert output.raw_deliverables["memo.md"] == (tasks_root / "contracts/t1/documents/notes.txt").read_bytes()


def test_forward_records_abandonment(tasks_root: Path) -> None:
    client = _CapturingClient(
        [
            ("code_exec", {"cmd": "printf 'unsubmitted' > memo.md"}),
            ("abandon_task_finish", {"reason": "inputs are missing"}),
        ]
    )
    output = _run_with_client(tasks_root, client)
    assert output.abandoned is True
    assert output.final_answer == "inputs are missing"
    assert output.deliverables == {}
    assert output.raw_deliverables == {}
    assert output.missing_deliverables == ["memo.md"]


def test_forward_does_not_grade_unsubmitted_files(tasks_root: Path) -> None:
    client = _CapturingClient(
        [
            ("code_exec", {"cmd": "printf 'unsubmitted' > memo.md"}),
            ("finish", {"summary": "done", "paths": []}),
        ]
    )
    output = _run_with_client(tasks_root, client)
    assert output.submitted_paths == []
    assert output.deliverables == {}
    assert output.raw_deliverables == {}
    assert output.missing_deliverables == ["memo.md"]


def test_forward_does_not_grade_files_after_max_turns(tasks_root: Path) -> None:
    client = _CapturingClient([("code_exec", {"cmd": "printf 'unfinished' > memo.md"})])
    output = _run_with_client(tasks_root, client)
    assert output.final_answer == ""
    assert output.submitted_paths == []
    assert output.deliverables == {}
    assert output.raw_deliverables == {}
    assert output.missing_deliverables == ["memo.md"]
    assert output.finished is False
    assert output.abandoned is False
    assert output.max_turns_reached is True
    assert output.total_turns == 4


def test_count_turns_counts_messages_across_compacted_history() -> None:
    from stirrup.core.models import AssistantMessage, UserMessage

    history = [
        [UserMessage(content="task"), AssistantMessage(content="one"), AssistantMessage(content="two")],
        [UserMessage(content="summary"), AssistantMessage(content="three")],
    ]
    assert _count_turns(history) == 3


def test_forward_collects_response_md_when_task_names_no_deliverable(tmp_path: Path) -> None:
    """A task with no declared deliverables defaults to `response.md` — and that
    fallback must drive collection + missing, not just the prompt, so a produced
    `response.md` is graded rather than silently dropped."""
    root = tmp_path / "tasks"
    _write_task(
        root,
        "contracts/freeform",
        criteria=[{"id": "C1", "title": "t", "match_criteria": "m", "deliverables": []}],
        deliverables={},
    )
    agent = HarveyLabAgent(
        config=HarveyLabConfig(max_turns=4, enable_view_image=False),
        task_source=task_source_from_dir(root),
        model_factory=lambda *_: _CapturingClient(
            [
                ("code_exec", {"cmd": "printf 'the answer' > response.md"}),
                ("finish", {"summary": "done", "paths": ["response.md"]}),
            ]
        ),
        exec_provider_factory=_local_exec_factory,
    )
    record = load_records(root, task_ids=["contracts/freeform"])[0]
    output = asyncio.run(agent.forward(record=record))
    assert output.deliverables == {"response.md": "the answer"}
    assert output.raw_deliverables == {"response.md": b"the answer"}
    assert output.missing_deliverables == []


def test_finish_rejects_relative_paths_and_stat_errors() -> None:
    class _BrokenEnv:
        async def file_exists(self, _path: str) -> bool:
            raise RuntimeError("stat failed")

        async def is_directory(self, _path: str) -> bool:
            return False

    async def _run(path: str) -> Any:
        finish = _build_finish_tools(_BrokenEnv())[0]
        result = finish.executor(FinishParams(summary="done", paths=[path]))
        return await result if inspect.isawaitable(result) else result

    assert asyncio.run(_run("memo.md")).success is False
    assert asyncio.run(_run("/tmp/memo.md")).success is False


def test_finish_accepts_posix_absolute_paths_cross_platform() -> None:
    """A POSIX container path (e.g. `/workspace/memo.md`) must count as absolute
    even on a Windows host, so a swapped-in container/remote backend still works.
    Here the path is absolute + stats as a real file, so finish succeeds."""

    class _Env:
        async def file_exists(self, _path: str) -> bool:
            return True

        async def is_directory(self, _path: str) -> bool:
            return False

    finish = _build_finish_tools(_Env())[0]
    result = finish.executor(FinishParams(summary="done", paths=["/workspace/memo.md"]))
    result = asyncio.run(result) if inspect.isawaitable(result) else result
    assert result.success is not False


def test_finish_rejects_paths_that_are_not_files(tasks_root: Path) -> None:
    """A submitted path that does not resolve to a real file is rejected, and the
    agent gets another turn (AA validates finish paths)."""
    client = _CapturingClient(
        [
            ("finish", {"summary": "premature", "paths": ["memo.md"]}),
            ("code_exec", {"cmd": "printf 'fee $50,000' > memo.md"}),
            ("finish", {"summary": "now for real", "paths": ["memo.md"]}),
        ]
    )
    output = _run_with_client(tasks_root, client)
    assert output.final_answer == "now for real"
    assert "memo.md" in output.deliverables


def test_forward_ignores_near_miss_filenames(tasks_root: Path) -> None:
    """A deliverable saved under the wrong filename counts as not produced."""
    agent = _build_agent(tasks_root, "Termination fee is $50,000.", name="my-memo.md")
    record = load_records(tasks_root, task_ids=["contracts/t1"])[0]
    output = asyncio.run(agent.forward(record=record))
    assert output.deliverables == {}
    assert output.missing_deliverables == ["memo.md"]


def test_forward_succeeds_in_worker_thread(tasks_root: Path) -> None:
    """The eval may run cases off the main thread; the Stirrup session must not
    install a SIGINT handler (would raise "signal only works in main thread of
    the main interpreter")."""
    import threading

    agent = _build_agent(tasks_root, "Termination fee is $50,000, per notes.txt.")
    record = load_records(tasks_root, task_ids=["contracts/t1"])[0]
    outcome: dict[str, Any] = {}

    def _worker() -> None:
        try:
            outcome["result"] = asyncio.run(evaluate_record(record=record, agent=agent, judge=_fee_judge))
        except BaseException as exc:  # noqa: BLE001 - surface to the assertion below
            outcome["error"] = exc

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()

    assert "error" not in outcome, f"evaluate_record raised off the main thread: {outcome.get('error')!r}"
    assert outcome["result"][ALL_PASS_FIELD] == 1.0


def test_evaluate_agent_on_records_aggregates(tasks_root: Path) -> None:
    agent = _build_agent(tasks_root, "Termination fee is $50,000, per notes.txt.")
    records = load_records(tasks_root)
    report = asyncio.run(evaluate_agent_on_records(agent=agent, records=records, judge=_fee_judge, max_concurrency=2))
    assert report.num_cases == 6
    assert report.num_scored == 6
    assert report.num_errored == 0
    assert report.all_pass_rate == 1.0
    assert report.criterion_pass_rate == 1.0


def test_evaluate_persists_reuses_and_reruns_outputs(
    tasks_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = load_records(tasks_root, task_ids=["contracts/t1"])[0]
    output_dir = tmp_path / "outputs"

    class _FakeAgent:
        calls = 0

        async def forward(self, *, record: Any) -> HarveyLabAgentOutput:
            self.calls += 1
            return HarveyLabAgentOutput(
                final_answer="done",
                deliverables={"memo.md": "not the persisted content"},
                raw_deliverables={"memo.md": b"Termination fee is $50,000."},
                missing_deliverables=[],
                finished=True,
                total_turns=2,
            )

    agent = _FakeAgent()
    monkeypatch.setattr(cli_mod, "_select_records", lambda _args: (tasks_root, [record]))
    monkeypatch.setattr(cli_mod, "_build_agent", lambda _root, _config: agent)
    monkeypatch.setattr(cli_mod, "build_rubric_judge", lambda **_kwargs: _fee_judge)
    args = cli_mod._parse_args(["evaluate", "--output-dir", str(output_dir), "--no-view-image"])

    assert cli_mod._run_evaluate(args) == 0
    assert agent.calls == 1
    assert (output_dir / "contracts/t1/memo.md").read_bytes() == b"Termination fee is $50,000."
    assert json.loads((output_dir / "eval_summary.json").read_text())["all_pass_rate"] == 1.0

    assert cli_mod._run_evaluate(args) == 0
    assert agent.calls == 1

    (output_dir / "contracts/t1/memo.md").unlink()
    assert cli_mod._run_evaluate(args) == 0
    assert agent.calls == 2

    args.rerun = True
    assert cli_mod._run_evaluate(args) == 0
    assert agent.calls == 3


def test_completed_empty_submission_is_reusable(tasks_root: Path, tmp_path: Path) -> None:
    record = load_records(tasks_root, task_ids=["contracts/t1"])[0]
    entry = {
        "task_id": record.task_id,
        "task_fingerprint": record.task_fingerprint,
        "deliverables_produced": [],
        "deliverables_missing": ["memo.md"],
        "abandoned": True,
    }
    assert cli_mod._is_reusable(tmp_path, record, entry)


def test_persisted_rerun_removes_stale_deliverables(tasks_root: Path, tmp_path: Path) -> None:
    """A re-run that produces nothing must clear a prior run's files — even when
    the intervening manifest entry was an error (carrying no produced list)."""
    record = load_records(tasks_root, task_ids=["contracts/t1"])[0]
    stale = tmp_path / "contracts/t1/memo.md"
    stale.parent.mkdir(parents=True)
    stale.write_text("old run")
    output = HarveyLabAgentOutput(final_answer="none", missing_deliverables=["memo.md"], finished=True)

    entry = cli_mod._persist_agent_output(tmp_path, record, output)

    assert not stale.exists()
    assert entry["deliverables_produced"] == []
    assert cli_mod._is_reusable(tmp_path, record, entry)


def test_max_turn_and_stale_task_entries_are_not_reusable(tasks_root: Path, tmp_path: Path) -> None:
    record = load_records(tasks_root, task_ids=["contracts/t1"])[0]
    entry = {
        "task_id": record.task_id,
        "task_fingerprint": record.task_fingerprint,
        "deliverables_produced": [],
        "deliverables_missing": ["memo.md"],
        "finished": False,
        "abandoned": False,
        "max_turns_reached": True,
    }
    assert not cli_mod._is_reusable(tmp_path, record, entry)

    entry.update(finished=True, max_turns_reached=False, task_fingerprint="stale")
    assert not cli_mod._is_reusable(tmp_path, record, entry)


def test_max_turn_run_is_graded_not_errored(tasks_root: Path, tmp_path: Path) -> None:
    """A max-turns run is not *reusable* (it gets re-run), but once persisted it
    must still be graded on whatever partial output it submitted — not bucketed
    as an error, which would drop its passes and inflate ``num_errored``."""
    record = load_records(tasks_root, task_ids=["contracts/t1"])[0]
    memo = tmp_path / "contracts/t1/memo.md"
    memo.parent.mkdir(parents=True)
    memo.write_text("partial memo body")
    entry = {
        "task_id": record.task_id,
        "task_fingerprint": record.task_fingerprint,
        "deliverables_produced": ["memo.md"],
        "deliverables_missing": [],
        "finished": False,
        "abandoned": False,
        "max_turns_reached": True,
        "final_answer": "ran out of turns",
    }

    assert not cli_mod._is_reusable(tmp_path, record, entry)
    assert cli_mod._is_gradeable(tmp_path, record, entry)

    outputs, errors = cli_mod._load_persisted_outputs(tmp_path, [record], {record.task_id: entry})
    assert errors == {}
    graded = outputs[record.task_id]
    assert graded.max_turns_reached is True
    assert graded.deliverables == {"memo.md": "partial memo body"}


def test_judge_failure_aborts_evaluation(tasks_root: Path) -> None:
    records = load_records(tasks_root, task_ids=["contracts/t1", "tax/t1"])
    outputs = {
        record.task_id: HarveyLabAgentOutput(
            final_answer="done", deliverables={"memo.md": "$50,000 in notes.txt"}, finished=True
        )
        for record in records
    }

    def _mixed_judge(desc: str, criteria: list[dict], _output: str) -> dict[str, bool]:
        if "tax/t1" in desc:
            raise JudgeCallError("context window exceeded")
        return {str(criterion["id"]): True for criterion in criteria}

    with pytest.raises(JudgeCallError, match="context window exceeded"):
        asyncio.run(evaluate_outputs_on_records(records=records, outputs=outputs, errors={}, judge=_mixed_judge))


def test_cli_judge_failure_returns_nonzero_without_partial_reports(
    tasks_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = load_records(tasks_root, task_ids=["contracts/t1"])[0]
    output_dir = tmp_path / "outputs"

    class _FakeAgent:
        async def forward(self, *, record: Any) -> HarveyLabAgentOutput:
            return HarveyLabAgentOutput(
                final_answer="done",
                deliverables={"memo.md": "body"},
                raw_deliverables={"memo.md": b"body"},
                finished=True,
            )

    def _broken_judge(_desc: str, _criteria: list[dict], _output: str) -> dict[str, bool]:
        raise JudgeCallError("context window exceeded")

    output_dir.mkdir()
    (output_dir / "eval_summary.json").write_text("stale")
    (output_dir / "eval_outputs.json").write_text("stale")
    monkeypatch.setattr(cli_mod, "_select_records", lambda _args: (tasks_root, [record]))
    monkeypatch.setattr(cli_mod, "_build_agent", lambda _root, _config: _FakeAgent())
    monkeypatch.setattr(cli_mod, "build_rubric_judge", lambda **_kwargs: _broken_judge)
    args = cli_mod._parse_args(["evaluate", "--output-dir", str(output_dir), "--no-view-image"])

    assert cli_mod._run_evaluate(args) == 1
    assert (output_dir / "run_outputs.json").is_file()
    assert (output_dir / "contracts/t1/memo.md").read_bytes() == b"body"
    assert not (output_dir / "eval_summary.json").exists()
    assert not (output_dir / "eval_outputs.json").exists()
    assert (output_dir / "eval_summary.previous.json").read_text() == "stale"
    assert (output_dir / "eval_outputs.previous.json").read_text() == "stale"


def test_evaluate_agent_contains_errors_and_excludes_unscoreable(tasks_root: Path) -> None:
    """One case erroring counts as 0 (deflates); an unscoreable case (empty
    rubric) is excluded from the denominator, not counted as a failure."""
    # Add an unscoreable task (no criteria).
    _write_task(tasks_root, "immigration/t1", criteria=[], deliverables={"memo.md": "Memo"})
    records = load_records(tasks_root)

    class _FlakyAgent(HarveyLabAgent):
        async def forward(self, *, record: Any) -> Any:  # type: ignore[override]
            if record.practice_area == "tax":
                raise RuntimeError("boom")  # forces that case to error
            return await super().forward(record=record)

    flaky = _FlakyAgent(
        config=HarveyLabConfig(max_turns=5, enable_view_image=False),
        task_source=task_source_from_dir(tasks_root),
        model_factory=_scripted_model_factory("Termination fee is $50,000, per notes.txt."),
        exec_provider_factory=_local_exec_factory,
    )
    report = asyncio.run(evaluate_agent_on_records(agent=flaky, records=records, judge=_fee_judge, max_concurrency=2))
    assert report.num_cases == 7
    assert report.num_errored == 1  # the tax task
    assert report.num_unscoreable == 1  # the immigration task (empty rubric)
    assert report.num_scored == 5
    # denominator = scored (5) + errored (1) = 6; the errored case scores 0.
    assert report.all_pass_rate == pytest.approx(5 / 6)
