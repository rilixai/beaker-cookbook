"""fs_search / fs_read / python_exec: capabilities, confinement, sandbox."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

from officeqa.agent.tools import PathOutsideWorkspace, PythonRepl, build_toolset, confine
from officeqa.data import corpus as corpus_mod
from officeqa.data.corpus import Workspace, create_workspace


def run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


@pytest.fixture
def tools(workspace: Workspace):  # type: ignore[no-untyped-def]
    ts = build_toolset(workspace, ("fs", "repl"))
    yield ts
    ts.close()


# --- confinement ------------------------------------------------------------


def test_confine_allows_cwd_and_corpus(workspace: Workspace) -> None:
    assert confine(".", workspace) == workspace.cwd.resolve()
    assert confine("../officeqa_corpus", workspace) == workspace.corpus_link.resolve()
    parsed = confine("../officeqa_corpus/parsed/treasury_bulletin_2013_03.txt", workspace)
    assert parsed.is_file()


def test_confine_hides_undeclared_files_in_the_corpus_root(workspace: Workspace, corpus: Path) -> None:
    """Only pdfs/ and parsed/ are reachable under ../officeqa_corpus, not whatever else lands in the cache root."""
    (corpus / "stray-credentials.json").write_text('{"token": "hf_secret"}')
    (corpus / "notes").mkdir()
    (corpus / "notes" / "x.txt").write_text("hf_secret")
    (corpus / "escape").symlink_to(Path.home())
    ts = build_toolset(workspace, ("fs",))
    for bad in ("../officeqa_corpus/stray-credentials.json", "../officeqa_corpus/notes", "../officeqa_corpus/escape"):
        with pytest.raises(PathOutsideWorkspace):
            confine(bad, workspace)
        assert "outside the working directory" in run(ts.call("fs_read", {"path": bad}))
    listing = run(ts.call("fs_search", {"path": "../officeqa_corpus"}))
    assert "pdfs/" in listing and "parsed/" in listing
    assert "stray" not in listing and "notes" not in listing and "escape" not in listing
    assert "hf_secret" not in run(ts.call("fs_search", {"path": "../officeqa_corpus", "query": "hf_secret"}))
    assert "stray" not in run(ts.call("fs_search", {"path": "../officeqa_corpus", "glob": "*"}))


@pytest.mark.parametrize("bad", ["..", "../..", "../../..", "/etc/passwd", "../.venv", "/", "../cwd/../.venv"])
def test_confine_rejects_outside(workspace: Workspace, bad: str) -> None:
    with pytest.raises(PathOutsideWorkspace):
        confine(bad, workspace)


def test_confine_rejects_symlink_escape(workspace: Workspace, tmp_path: Path) -> None:
    outside = tmp_path / "secret.txt"
    outside.write_text("nope")
    (workspace.cwd / "sneaky").symlink_to(outside)
    with pytest.raises(PathOutsideWorkspace):
        confine("sneaky", workspace)


def test_fs_tools_return_errors_not_content_outside(tools) -> None:  # type: ignore[no-untyped-def]
    out = run(tools.call("fs_read", {"path": "/etc/passwd"}))
    assert out.startswith("Error:") and "root:" not in out
    out = run(tools.call("fs_search", {"path": "../.."}))
    assert out.startswith("Error:")


# --- fs_search --------------------------------------------------------------


def test_fs_search_lists_directory(tools) -> None:  # type: ignore[no-untyped-def]
    out = run(tools.call("fs_search", {"path": "../officeqa_corpus"}))
    assert "../officeqa_corpus/pdfs/" in out and "../officeqa_corpus/parsed/" in out
    out = run(tools.call("fs_search", {"path": "../officeqa_corpus/parsed"}))
    assert out.count("treasury_bulletin_") == 3


def test_fs_search_lists_by_pattern(tools) -> None:  # type: ignore[no-untyped-def]
    out = run(tools.call("fs_search", {"path": "../officeqa_corpus/parsed", "glob": "*2013*.txt"}))
    assert "treasury_bulletin_2013_03.txt" in out and "treasury_bulletin_2013_06.txt" in out
    assert "1987" not in out


def test_fs_search_greps_recursively_with_context(tools) -> None:  # type: ignore[no-untyped-def]
    out = run(
        tools.call(
            "fs_search", {"path": "../officeqa_corpus/parsed", "query": "massachusetts", "before": 1, "after": 1}
        )
    )
    assert "treasury_bulletin_2013_03.txt:5:| Massachusetts | 0.866 |" in out
    assert "| --- | --- |" in out  # -B 1
    assert "Texas" in out  # -A 1


def test_fs_search_in_specific_files(tools) -> None:  # type: ignore[no-untyped-def]
    out = run(
        tools.call(
            "fs_search",
            {
                "path": "../officeqa_corpus/parsed",
                "glob": "treasury_bulletin_2013*.txt",
                "query": "Individual income taxes",
            },
        )
    )
    assert "2013_06" in out and "1987" not in out
    out = run(
        tools.call(
            "fs_search", {"path": "../officeqa_corpus/parsed/treasury_bulletin_1987_06.txt", "query": "total receipts"}
        )
    )
    assert "9876543.21" in out


def test_fs_search_skips_pdfs_and_caps_results(tools) -> None:  # type: ignore[no-untyped-def]
    out = run(tools.call("fs_search", {"path": "../officeqa_corpus/pdfs", "query": "Massachusetts"}))
    assert "0.866" not in out
    out = run(tools.call("fs_search", {"path": "../officeqa_corpus/parsed", "query": "filler line", "max_results": 5}))
    assert out.count("filler line") == 5


# --- fs_read ----------------------------------------------------------------


def test_fs_read_head_slice_whole(tools) -> None:  # type: ignore[no-untyped-def]
    p = "../officeqa_corpus/parsed/treasury_bulletin_2013_03.txt"
    head = run(tools.call("fs_read", {"path": p, "head": 2}))
    assert "lines 1-2 of" in head and "Massachusetts" not in head
    sl = run(tools.call("fs_read", {"path": p, "start": 5, "end": 6}))
    assert "5\t| Massachusetts | 0.866 |" in sl and "6\t| Texas" in sl and "State" not in sl
    whole = run(tools.call("fs_read", {"path": p}))
    assert "Massachusetts" in whole and "Texas" in whole


def test_fs_read_refuses_oversized_whole_file(workspace: Workspace) -> None:
    ts = build_toolset(workspace, ("fs",), output_limit=500)
    out = run(ts.call("fs_read", {"path": "../officeqa_corpus/parsed/treasury_bulletin_1987_06.txt"}))
    assert out.startswith("Error:") and "exceeds the 500-character read limit" in out
    assert "filler line 100" not in out
    sl = run(
        ts.call("fs_read", {"path": "../officeqa_corpus/parsed/treasury_bulletin_1987_06.txt", "start": 1, "end": 5})
    )
    assert "9876543.21" in sl


def test_fs_read_refuses_binary_pdf(tools) -> None:  # type: ignore[no-untyped-def]
    out = run(tools.call("fs_read", {"path": "../officeqa_corpus/pdfs/treasury_bulletin_2013_03.pdf"}))
    assert out.startswith("Error:") and "python_exec" in out and "Massachusetts" not in out


def test_fs_read_can_read_files_written_in_cwd(tools, workspace: Workspace) -> None:  # type: ignore[no-untyped-def]
    (workspace.cwd / "notes.txt").write_text("a\nb\nc\n")
    assert "2\tb" in run(tools.call("fs_read", {"path": "notes.txt"}))


# --- python_exec ------------------------------------------------------------


def test_python_exec_is_stateful(tools) -> None:  # type: ignore[no-untyped-def]
    assert run(tools.call("python_exec", {"code": "x = 21"})) == "(no output)"
    assert run(tools.call("python_exec", {"code": "print(x * 2)"})).strip() == "42"
    assert run(tools.call("python_exec", {"code": "x + 1"})).strip() == "22"  # REPL-style echo


def test_python_exec_does_not_inherit_host_secrets(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-for-the-model")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    ts = build_toolset(workspace, ("repl",))
    try:
        out = run(
            ts.call(
                "python_exec",
                {"code": "import os; print(sorted(k for k in os.environ if 'KEY' in k or 'TOKEN' in k))"},
            )
        )
        assert out.strip() == "[]"
        assert "sk-test" not in run(ts.call("python_exec", {"code": "import os; print(dict(os.environ))"}))
        assert run(ts.call("python_exec", {"code": "import os; print('PATH' in os.environ)"})).strip() == "True"
    finally:
        ts.close()


def test_isolated_python_exec_sees_recipe_env_packages(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Packages installed in the harness's own venv (install-ocr-deps.sh) must import in the per-question venv."""
    parent_site = tmp_path / "recipe-site-packages"
    parent_site.mkdir()
    (parent_site / "officeqa_probe_ocr_lib.py").write_text("VERSION = 'from-recipe-venv'\n")
    monkeypatch.setattr(corpus_mod, "parent_site_packages", lambda: [str(parent_site)])
    ws = create_workspace(tmp_path / "work", "q-venv", corpus, isolate=True)
    assert ws.python != sys.executable
    ts = build_toolset(ws, ("repl",))
    try:
        assert run(ts.call("python_exec", {"code": "import sys; print(sys.prefix)"})).strip() == str(ws.root / ".venv")
        out = run(ts.call("python_exec", {"code": "import officeqa_probe_ocr_lib as m; print(m.VERSION)"}))
        assert out.strip() == "from-recipe-venv"
        # The child's own site-packages come first, so agent installs shadow the parent and stay local.
        code = (
            "import sys, sysconfig; own = sysconfig.get_paths()['purelib']; "
            f"print(sys.path.index(own) < sys.path.index({str(parent_site)!r}))"
        )
        assert run(ts.call("python_exec", {"code": code})).strip() == "True"
    finally:
        ts.close()


def test_python_exec_captures_stderr_and_tracebacks(tools) -> None:  # type: ignore[no-untyped-def]
    out = run(tools.call("python_exec", {"code": "1/0"}))
    assert "ZeroDivisionError" in out


def test_python_exec_sees_corpus_relative_to_cwd(tools) -> None:  # type: ignore[no-untyped-def]
    out = run(
        tools.call(
            "python_exec",
            {"code": "print(open('../officeqa_corpus/parsed/treasury_bulletin_2013_03.txt').read()[:26])"},
        )
    )
    assert out.strip() == "Treasury Bulletin March 20"


@pytest.mark.parametrize(
    "code",
    [
        "import glob; glob.glob('../officeqa_corpus/parsed/*')",
        "import glob; list(glob.iglob('*'))",
        "import os; os.listdir('../officeqa_corpus/parsed')",
        "import os; list(os.walk('../officeqa_corpus'))",
        "import os; list(os.scandir('.'))",
        "from pathlib import Path; list(Path('../officeqa_corpus/parsed').glob('*.txt'))",
        "from pathlib import Path; list(Path('..').rglob('*'))",
        "from pathlib import Path; list(Path('.').iterdir())",
        "import subprocess; subprocess.run(['ls', '../officeqa_corpus/parsed'])",
        "import subprocess; subprocess.check_output('find ../officeqa_corpus -name \"*.txt\"', shell=True)",
    ],
)
def test_python_exec_blocks_bulk_enumeration(tools, code: str) -> None:  # type: ignore[no-untyped-def]
    out = run(tools.call("python_exec", {"code": code}))
    assert "PermissionError" in out and "fs_search" in out
    assert "treasury_bulletin_1987_06" not in out


def test_python_exec_allows_other_subprocesses(tools) -> None:  # type: ignore[no-untyped-def]
    out = run(
        tools.call(
            "python_exec",
            {
                "code": "import subprocess; print(subprocess.run(['echo', 'hi'], capture_output=True, text=True).stdout)"
            },
        )
    )
    assert out.strip() == "hi"


def test_python_exec_timeout_kills_and_restarts(workspace: Workspace) -> None:
    repl = PythonRepl(workspace, default_timeout_s=1.0)
    try:
        run(repl.run("y = 5"))
        out = run(repl.run("import time; time.sleep(30)"))
        assert out.startswith("Error:") and "killed" in out
        out = run(repl.run("print('y' in dir())"))
        assert out.strip() == "False"  # state lost after kill, process restarted
    finally:
        repl.close()


SPAWN_SLOW_WRITER = (
    "import subprocess, sys; "
    "p = subprocess.Popen([sys.executable, '-c', "
    "\"import time; time.sleep(2); open('child_out.txt', 'w').write('escaped')\"]); "
    "open('child.pid', 'w').write(str(p.pid))"
)


def _pid_alive(pid: int) -> bool:
    """Running (a killed-but-unreaped zombie counts as dead)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status = Path(f"/proc/{pid}/status")
    if status.exists():
        return "State:\tZ" not in status.read_text()
    return True


def _child_pid(workspace: Workspace) -> int:
    return int((workspace.cwd / "child.pid").read_text())


def test_repl_timeout_kills_subprocesses_spawned_by_agent_code(workspace: Workspace) -> None:
    """The timeout bounds everything the question launched, not just the REPL interpreter."""
    repl = PythonRepl(workspace, default_timeout_s=1.0)
    try:
        assert run(repl.run(SPAWN_SLOW_WRITER)) == "(no output)"
        child = _child_pid(workspace)
        assert _pid_alive(child)
        out = run(repl.run("import time; time.sleep(30)"))
        assert "killed" in out
        time.sleep(0.5)
        assert not _pid_alive(child)
        time.sleep(2.5)
        assert not (workspace.cwd / "child_out.txt").exists()
    finally:
        repl.close()


def test_repl_close_kills_orphans_of_an_already_exited_repl(workspace: Workspace) -> None:
    repl = PythonRepl(workspace)
    try:
        out = run(repl.run(SPAWN_SLOW_WRITER + "; import os; os._exit(0)"))
        assert "REPL process exited" in out
        child = _child_pid(workspace)
        time.sleep(0.5)
        assert not _pid_alive(child)
    finally:
        repl.close()

    repl = PythonRepl(workspace)
    assert run(repl.run(SPAWN_SLOW_WRITER)) == "(no output)"
    child = _child_pid(workspace)
    repl.close()
    time.sleep(0.5)
    assert not _pid_alive(child)
    time.sleep(2.5)
    assert not (workspace.cwd / "child_out.txt").exists()


def test_toolset_registration_and_unknown_tool(workspace: Workspace) -> None:
    ts = build_toolset(workspace, ("fs", "repl"))
    try:
        assert sorted(ts.tools) == ["fs_read", "fs_search", "python_exec"]
        assert all(s["type"] == "function" for s in ts.schemas())
        assert run(ts.call("web_search", {"query": "x"})).startswith("Error: unknown tool")
        assert run(ts.call("fs_read", {"nope": 1})).startswith("Error: bad arguments")
    finally:
        ts.close()


def test_web_search_only_when_requested(workspace: Workspace) -> None:
    class FakeBackend:
        def search(self, query: str, max_results: int) -> list[dict[str, str]]:
            return [{"title": "T", "url": "https://x", "snippet": query}]

    ts = build_toolset(workspace, ("fs", "web"), web_backend=FakeBackend())
    assert "web_search" in ts.tools and "python_exec" not in ts.tools
    assert "https://x" in run(ts.call("web_search", {"query": "gdp"}))
