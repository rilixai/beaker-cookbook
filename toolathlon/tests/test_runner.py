from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from toolathlon_baseline import runner


@pytest.fixture
def prepared_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate result handling from network downloads and benchmark contents."""
    source = tmp_path / "upstream"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts/formal_run_v0.json").write_text(json.dumps({"global_task_config": {}, "agent": {}, "user": {}}))
    monkeypatch.setenv("TOOLATHLON_OPENAI_BASE_URL", "http://localhost:12345/v1")
    monkeypatch.setattr(runner, "prepared_source", lambda cache: source)

    def stage(source: Path, cache: Path, output: Path, task: str, system_prompt: Path | None) -> Path:
        return output

    monkeypatch.setattr(runner, "stage_task", stage)
    return tmp_path / "run"


@pytest.mark.parametrize(
    ("verdict", "code", "expected_status"),
    [(True, 0, "passed"), (True, 1, "error"), (False, 1, "failed"), (None, 1, "error"), ("false", 0, "invalid")],
)
def test_preserves_official_verdict(
    prepared_run: Path, monkeypatch: pytest.MonkeyPatch, verdict: object, code: int, expected_status: str
) -> None:
    def execute(*args: object, **kwargs: object) -> int:
        artifacts = prepared_run / "artifacts"
        artifacts.mkdir()
        (artifacts / "eval_res.json").write_text(json.dumps({"pass": verdict, "details": "official diagnostics"}))
        return code

    monkeypatch.setattr(runner, "execute", execute)
    if expected_status == "invalid":
        with pytest.raises(ValueError, match="invalid pass/fail"):
            runner.run_task(cache=prepared_run.parent, output=prepared_run, task="find-alita-paper", model="model")
        assert json.loads((prepared_run / "result.json").read_text())["status"] == "error"
    else:
        result = runner.run_task(
            cache=prepared_run.parent, output=prepared_run, task="find-alita-paper", model="model"
        )
        assert result["status"] == expected_status
        assert result["pass"] is verdict
        assert result["evaluation"]["details"] == "official diagnostics"


def test_missing_evaluation_is_unscored(prepared_run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "execute", lambda *args, **kwargs: 0)
    result = runner.run_task(cache=prepared_run.parent, output=prepared_run, task="find-alita-paper", model="model")
    assert result["status"] == "error"
    assert result["pass"] is None


def test_never_reuses_previous_output(prepared_run: Path) -> None:
    prepared_run.mkdir()
    previous = prepared_run / "result.json"
    previous.write_text("previous run\n")
    with pytest.raises(FileExistsError):
        runner.run_task(cache=prepared_run.parent, output=prepared_run, task="find-alita-paper", model="model")
    assert previous.read_text() == "previous run\n"


def test_timeout_keeps_summary(prepared_run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def execute(*args: object, **kwargs: object) -> int:
        raise subprocess.TimeoutExpired("agent", 1)

    monkeypatch.setattr(runner, "execute", execute)
    result = runner.run_task(cache=prepared_run.parent, output=prepared_run, task="find-alita-paper", model="model")
    assert result["status"] == "timeout"
    assert result["pass"] is None
    assert json.loads((prepared_run / "result.json").read_text()) == result


def test_timeout_stops_mcp_descendant(tmp_path: Path) -> None:
    # A child which ignores TERM reproduces a stuck tool server. The parent
    # must be reaped and the child must not continue after the task timeout.
    pid_file = tmp_path / "tool.pid"
    child = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    parent = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"p=subprocess.Popen([sys.executable,'-c',{child!r}]); "
        f"Path({str(pid_file)!r}).write_text(str(p.pid)); time.sleep(60)"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        runner.execute(
            [sys.executable, "-c", parent], cwd=tmp_path, env=os.environ.copy(), log=tmp_path / "run.log", timeout=2
        )
    pid = int(pid_file.read_text())
    # On Linux an orphan may briefly remain as a zombie until the host reaps
    # it; a zombie is terminated and cannot execute or retain MCP resources.
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text().split()[2] == "Z":
            return
        time.sleep(0.01)
    os.kill(pid, signal.SIGKILL)
    pytest.fail("The tool server survived its task timeout")
