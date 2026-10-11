"""Corpus manifest, cache wiring, CLI resource check, and prompt variants."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from officeqa import config
from officeqa.agent import prompts
from officeqa.cli import log_resources
from officeqa.config import RunConfig
from officeqa.data.corpus import IncompleteCorpusError, count_documents, ensure_corpus, fetch_corpus
from officeqa.data.manifest import build_manifest


def test_cache_dir_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OFFICEQA_CACHE_DIR", str(tmp_path / "c"))
    assert config.cache_dir() == tmp_path / "c"
    monkeypatch.delenv("OFFICEQA_CACHE_DIR")
    assert config.cache_dir() == Path.home() / ".cache" / "officeqa"


def test_manifest_paths_and_representations(corpus: Path) -> None:
    m = build_manifest(corpus)
    assert m.root == "../officeqa_corpus/" and m.default == "pdfs"  # agent-relative: no host paths in traces
    by_name = {r.name: r for r in m.representations}
    assert set(by_name) == {"pdfs", "parsed"}
    assert by_name["pdfs"].format == "pdf" and by_name["parsed"].format == "text"
    assert by_name["pdfs"].documents == 3 and by_name["parsed"].documents == 3
    assert by_name["pdfs"].path == "pdfs/" and by_name["parsed"].path == "parsed/"
    assert str(corpus) not in json.dumps(m.to_json())
    assert build_manifest(corpus, default="parsed").default == "parsed"
    json.dumps(m.to_json())


def test_ensure_corpus_requires_representation(fake_cache: Path) -> None:
    assert ensure_corpus("pdfs") == fake_cache / config.CORPUS_ROOT_NAME
    (fake_cache / config.CORPUS_ROOT_NAME / "parsed").unlink()
    shutil.rmtree(fake_cache / "hf" / config.CORPUS_REPRESENTATIONS["parsed"][0])
    with pytest.raises(FileNotFoundError):
        ensure_corpus("parsed")


def test_ensure_corpus_rejects_incomplete_download(fake_cache: Path) -> None:
    root = fake_cache / config.CORPUS_ROOT_NAME
    n = count_documents(root, "parsed")
    assert ensure_corpus("parsed", expected_documents=n) == root
    # An interrupted `officeqa fetch` leaves fewer documents than the pinned snapshot has.
    next(p for p in (root / "parsed").iterdir() if p.suffix == ".txt").unlink()
    with pytest.raises(IncompleteCorpusError, match=rf"has {n - 1} documents, expected {n}.*officeqa fetch"):
        ensure_corpus("parsed", expected_documents=n)
    assert ensure_corpus("parsed") == root  # library callers without an expectation are unaffected
    # Stray non-document files (e.g. partial-download markers) do not count as documents.
    (fake_cache / "hf" / config.CORPUS_REPRESENTATIONS["parsed"][0] / "x.txt.incomplete").write_bytes(b"")
    assert count_documents(root, "parsed") == n - 1


def test_fetch_corpus_uses_snapshot_and_symlinks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_snapshot(*, allow_patterns: list[str], local_dir: Path, revision: str) -> None:
        calls.append({"allow_patterns": allow_patterns, "revision": revision})
        for pat in allow_patterns:
            sub = Path(pat).parent
            d = local_dir / sub
            d.mkdir(parents=True, exist_ok=True)
            (d / "treasury_bulletin_2000_01.txt").write_text("x")

    monkeypatch.setattr("officeqa.data.corpus._snapshot", fake_snapshot)
    root = fetch_corpus(("parsed",), cache=tmp_path)
    assert calls == [{"allow_patterns": [config.CORPUS_PARSED_PATTERN], "revision": config.OFFICEQA_DATASET_REVISION}]
    assert (root / "parsed").is_symlink() and (root / "parsed" / "treasury_bulletin_2000_01.txt").is_file()
    assert not (root / "pdfs").exists()


def test_log_resources_warns_on_oversubscription(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("os.cpu_count", lambda: 4)
    assert log_resources(RunConfig(max_concurrent=16, corpus="pdfs"))["oversubscribed_warning"] is True
    assert log_resources(RunConfig(max_concurrent=16, corpus="parsed"))["oversubscribed_warning"] is False
    assert log_resources(RunConfig(max_concurrent=4, corpus="pdfs"))["oversubscribed_warning"] is False


def test_run_config_defaults_match_baseline() -> None:
    cfg = RunConfig()
    assert (cfg.tools, cfg.corpus, cfg.search, cfg.n_rollouts) == (("fs", "repl"), "pdfs", "fs", 1)
    assert (cfg.max_steps, cfg.window_size, cfg.tool_output_limit, cfg.max_retries) == (200, 30, 25_000, 30)
    assert (cfg.max_concurrent, cfg.task_timeout_s, cfg.reasoning_effort) == (16, 3600.0, "high")
    with pytest.raises(ValueError):
        RunConfig(tools=("fs", "laser"))


@pytest.mark.parametrize(
    "bad",
    [
        {"max_concurrent": 0},  # would block forever on the semaphore
        {"max_concurrent": -1},
        {"n_rollouts": 0},
        {"max_steps": 0},
        {"window_size": 0},
        {"tool_output_limit": 0},
        {"max_retries": -1},
        {"max_output_tokens": 0},
        {"task_timeout_s": 0.0},
        {"llm_timeout_s": -5.0},
    ],
)
def test_run_config_rejects_out_of_range_numbers(bad: dict[str, float]) -> None:
    (name,) = bad
    with pytest.raises(ValueError, match=name):
        RunConfig(**bad)  # type: ignore[arg-type]
    RunConfig(max_retries=0, max_concurrent=1, n_rollouts=1)  # boundaries are allowed


def test_run_config_behavior_excludes_operational_knobs() -> None:
    a = RunConfig()
    b = RunConfig(max_concurrent=1, max_retries=0, task_timeout_s=10.0, llm_timeout_s=10.0)
    assert a.behavior() == b.behavior()
    for change in ({"model": "other"}, {"tools": ("fs", "repl", "web")}, {"corpus": "parsed"}, {"max_steps": 10}):
        assert RunConfig(**change).behavior() != a.behavior()  # type: ignore[arg-type]


# --- prompts ------------------------------------------------------------------

REVISION_LINE = (
    "If multiple documents report the same metric, use the most up-to-date revision unless the question "
    "specifies an exact date or document. Do not rely on the first matching value you encounter."
)
WEB_SENTENCE = "You also have access to web search in the case that you need to look something up."


def test_seed_prompt_differs_only_by_web_sentence() -> None:
    v, s = prompts.SYSTEM_PROMPT_VERBATIM, prompts.SYSTEM_PROMPT_SEED
    assert WEB_SENTENCE in v and WEB_SENTENCE not in s
    assert REVISION_LINE in v and REVISION_LINE in s
    assert v.replace(" " + WEB_SENTENCE, "", 1) == s
    for text in (v, s):
        assert "../officeqa_corpus/" in text and "<FINAL_ANSWER>" in text and "FAILURE CONDITION" in text


def test_prompt_selection_and_reminder() -> None:
    assert prompts.system_prompt_for(("fs", "repl")) == prompts.SYSTEM_PROMPT_SEED
    assert prompts.system_prompt_for(("fs", "repl", "web")) == prompts.SYSTEM_PROMPT_VERBATIM
    # The agent passes registered tool names, not --tools names.
    assert prompts.system_prompt_for(("fs_search", "fs_read", "python_exec")) == prompts.SYSTEM_PROMPT_SEED
    assert prompts.system_prompt_for(("fs_search", "fs_read", "web_search")) == prompts.SYSTEM_PROMPT_VERBATIM
    assert "1 step" in prompts.step_reminder(1) or "last step" in prompts.step_reminder(1)
    assert "42 steps remaining" in prompts.step_reminder(42)


def test_reload_rebinds_entry_points() -> None:
    # In a subprocess: reload() swaps module objects, which would confuse
    # `pytest.raises(SomeClass)` in tests that imported the old classes.
    code = (
        "import officeqa\n"
        "before = officeqa.run_one\n"
        "officeqa.reload()\n"
        "assert officeqa.run_one is not before\n"
        "assert officeqa.run_one.__module__ == 'officeqa.runner'\n"
        "from officeqa.agent import prompts\n"
        "assert prompts.system_prompt_for(('fs', 'repl')) == prompts.SYSTEM_PROMPT_SEED\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
