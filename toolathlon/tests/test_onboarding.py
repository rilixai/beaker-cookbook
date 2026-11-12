from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".beaker"))
import beaker_integration as integration  # noqa: E402
import toolathlon_setup as setup  # noqa: E402
from toolathlon_rollout import stage_agent  # noqa: E402


def test_workspace_transfer_is_complete_or_fails(tmp_path: Path) -> None:
    (tmp_path / "a_debug.bin").write_bytes(b"x" * 8)
    (tmp_path / "segment_growth_rates.xlsx").write_bytes(b"workbook")
    with pytest.raises(ValueError, match="transfer limit"):
        setup.collect_workspace_files(tmp_path, max_total_bytes=8)
    files = setup.collect_workspace_files(tmp_path, max_total_bytes=16)
    assert base64.b64decode(files["segment_growth_rates.xlsx"]) == b"workbook"


def test_workspace_symlink_cannot_read_trusted_reference(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    reference = tmp_path / "reference.xlsx"
    reference.write_bytes(b"trusted answer")
    (workspace / "output.xlsx").symlink_to(reference)
    with pytest.raises(ValueError, match="symlinks"):
        setup.collect_workspace_files(workspace)


def test_unsupported_sandbox_fails_setup_without_unrestricted_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        setup.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 1, stdout="", stderr="Landlock unavailable"),
    )
    with pytest.raises(RuntimeError, match="Refusing unrestricted execution"):
        setup.verify_sandbox_support()


@pytest.mark.parametrize("status", ["max_turns_reached", "failed"])
def test_failed_model_turn_is_not_a_bootstrap_retry(status: str) -> None:
    assert integration._has_model_turn(status, {"agent_llm_requests": 0}, {"messages": [{"role": "tool"}]})
    assert integration._has_model_turn("max_turns_reached", {}, {})
    assert not integration._has_model_turn("failed", {}, {"messages": [{"role": "user"}]})


def test_zero_counter_turn_limit_is_scored_without_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from beaker import Case

    class Trace:
        @contextlib.contextmanager
        def stage(self, *args: object, **kwargs: object):
            yield self

        def output(self, value: object) -> None:
            pass

    class Runtime:
        trace = Trace()

    monkeypatch.setattr(integration, "prepare_toolathlon", lambda: (tmp_path, tmp_path))
    monkeypatch.setattr(integration, "prepare_agent_runtime", lambda cache, source: tmp_path)
    monkeypatch.setattr(integration, "_resolve_model_target", lambda runtime: ("http://local", "test", "test", False))
    monkeypatch.setattr(
        integration,
        "_SUBPROCESS_SCRIPT",
        "import json,sys; from pathlib import Path; "
        "Path(sys.argv[2]).write_text(json.dumps({'status':'error','agent_status':'max_turns_reached',"
        "'key_stats':{'agent_llm_requests':0}}))",
    )
    task = "excel-market-research"
    references = tmp_path / "tasks/finalpool" / task / "groundtruth_workspace"
    references.mkdir(parents=True)
    (references / "answer.xlsx").write_bytes(b"trusted")

    async def exercise() -> None:
        result = await integration.run_case(
            case_input={"task": task, "instruction": "task", "mcp_servers": ["excel"]}, runtime=Runtime()
        )
        case = Case(
            id=task,
            input={},
            expected={
                "task": task,
                "upstream_revision": "revision",
                "evaluation_module": "evaluation.main",
                "groundtruth_hashes": setup.groundtruth_hashes(tmp_path, task),
            },
        )
        score = await integration.score_case(case=case, result=result, case_files_dir=tmp_path)
        assert score.objective == 0.0

    asyncio.run(exercise())


def test_missing_context_does_not_become_a_zero_score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from beaker import Case, CaseResult, RetryableCaseError

    monkeypatch.setattr(integration, "prepare_toolathlon", lambda: (tmp_path, tmp_path))
    case = Case(
        id="excel-market-research",
        input={},
        expected={
            "task": "excel-market-research",
            "upstream_revision": "revision",
            "evaluation_module": "evaluation.main",
            "groundtruth_hashes": {"answer.xlsx": "hash"},
        },
    )
    with pytest.raises(RetryableCaseError, match="scoring context"):
        asyncio.run(integration.score_case(case=case, result=CaseResult(output={}), case_files_dir=tmp_path))


def test_staged_solver_has_inputs_but_no_answers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("toolathlon_rollout.patch_source", lambda checkout: None)
    source = tmp_path / "source"
    for directory in ("utils", "scripts", "configs/mcp_servers", "tasks/finalpool/find-alita-paper/docs"):
        (source / directory).mkdir(parents=True)
    for name in ("main.py", "pyproject.toml", "uv.lock"):
        (source / name).write_text("")
    for name in ("global_configs", "token_key_session"):
        (source / f"configs/{name}_example.py").write_text("")
    task = source / "tasks/finalpool/find-alita-paper"
    (task / "task_config.json").write_text(json.dumps({"needed_mcp_servers": []}))
    for name in ("initial_workspace", "evaluation", "groundtruth_workspace"):
        (task / name).mkdir()
        (task / name / "file").write_text(name)
    output = tmp_path / "run"
    output.mkdir()
    checkout = stage_agent(source, tmp_path, output, "find-alita-paper", None)
    staged = checkout / "tasks/finalpool/find-alita-paper"
    assert (staged / "initial_workspace/file").read_text() == "initial_workspace"
    assert not (staged / "groundtruth_workspace").exists()
    assert not (staged / "evaluation").exists()
    assert (task / "groundtruth_workspace/file").read_text() == "groundtruth_workspace"
    assert integration.integration.targets[0].paths == ("prompts/agent_system_prompt.md",)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process groups")
def test_cancel_reaps_supervisor_and_stubborn_mcp(tmp_path: Path) -> None:
    pid_file = tmp_path / "tool.pid"
    tool = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    supervisor = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"p=subprocess.Popen([sys.executable,'-c',{tool!r}]); "
        f"Path({str(pid_file)!r}).write_text(str(p.pid)); time.sleep(60)"
    )

    async def exercise() -> None:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            supervisor,
            start_new_session=True,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        task = asyncio.create_task(integration._communicate(process))
        try:
            for _ in range(200):
                if pid_file.exists():
                    break
                await asyncio.sleep(0.01)
            assert pid_file.exists()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert process.returncode is not None
            pid = int(pid_file.read_text())
            stat = Path(f"/proc/{pid}/stat")
            assert not stat.exists() or stat.read_text().split()[2] == "Z"
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()

    asyncio.run(exercise())


@pytest.mark.skipif(sys.platform != "linux", reason="Linux Landlock")
def test_sandbox_denies_reference_cache_and_inherits_to_tools(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    reference = tmp_path / "trusted-reference"
    reference.write_text("answer")
    (workspace / "escape").symlink_to(reference)
    script = """
import os, subprocess, sys
from pathlib import Path
from toolathlon_sandbox import restrict_filesystem
workspace, reference = map(Path, sys.argv[1:])
os.chdir(workspace)
restrict_filesystem(
    read_paths=[Path('/usr'), Path('/lib'), Path('/lib64'), Path(sys.base_prefix), Path(sys.prefix),
                Path('/dev/urandom'), Path('/dev/random')],
    write_paths=[workspace, Path('/dev/null')],
)
(workspace / 'output').write_text('legitimate result')
for path in (reference, workspace / 'escape'):
    try:
        path.read_text()
    except PermissionError:
        pass
    else:
        raise AssertionError('reference leaked')
child = subprocess.run([sys.executable, '-c', 'import sys; open(sys.argv[1]).read()', str(reference)],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
assert child.returncode != 0 and b'PermissionError' in child.stderr, child.stderr
"""
    env = os.environ | {"PYTHONPATH": str(Path(setup.__file__).parent)}
    proc = subprocess.run(
        [sys.executable, "-c", script, str(workspace), str(reference)], env=env, capture_output=True, text=True
    )
    # CI/container hosts must support Landlock to exercise the actual boundary;
    # unsupported kernels are still required to fail closed in production.
    if "Landlock sandbox unavailable" in proc.stderr:
        pytest.skip("Host kernel or container blocks Landlock; onboarding will fail closed")
    assert proc.returncode == 0, proc.stderr
    assert (workspace / "output").read_text() == "legitimate result"
