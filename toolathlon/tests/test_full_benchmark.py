from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from toolathlon_baseline import catalog, containerized


@pytest.fixture
def benchmark_source(tmp_path: Path) -> Path:
    source = tmp_path / "upstream"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts/formal_run_v0.json").write_text(json.dumps({"global_task_config": {}, "agent": {}, "user": {}}))
    # Small protocol fixtures exercise selection, not benchmark task quality.
    for name in ("ab-testing", "k8s-mysql", "find-alita-paper"):
        folder = source / "tasks/finalpool" / name
        (folder / "docs").mkdir(parents=True)
        (folder / "evaluation").mkdir()
        (folder / "task_config.json").write_text(json.dumps({"needed_mcp_servers": ["filesystem"]}))
        (folder / "docs/task.md").write_text("task protocol fixture")
        (folder / "evaluation/main.py").write_text("# evaluator protocol fixture")
    (source / "tasks/finalpool/__pycache__").mkdir()
    return source


def test_discovers_full_catalog_without_native_allowlist(benchmark_source: Path) -> None:
    tasks = catalog.task_catalog(benchmark_source)
    assert [t["id"] for t in tasks] == ["ab-testing", "find-alita-paper", "k8s-mysql"]
    assert catalog.select_tasks(benchmark_source, None) == [t["id"] for t in tasks]
    assert catalog.select_tasks(benchmark_source, ["k8s-mysql"]) == ["k8s-mysql"]
    assert tasks[0]["native_quickstart"] is False


@pytest.mark.parametrize("selection", [["does-not-exist"], ["../ab-testing"], [], ["ab-testing", "ab-testing"]])
def test_rejects_invalid_selection(benchmark_source: Path, selection: list[str]) -> None:
    with pytest.raises(ValueError):
        catalog.select_tasks(benchmark_source, selection)


def write_verdict(artifacts: Path, name: str, verdict: object) -> None:
    folder = artifacts / "finalpool" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "eval_res.json").write_text(json.dumps({"pass": verdict}))


def test_collection_keeps_every_requested_task(tmp_path: Path) -> None:
    for name, verdict in (("pass", True), ("fail", False), ("incomplete", None), ("invalid", "true")):
        write_verdict(tmp_path, name, verdict)
    results = containerized.collect_results(tmp_path, ["pass", "fail", "incomplete", "invalid", "missing"])
    assert [r["status"] for r in results.values()] == ["passed", "failed", "error", "error", "error"]
    assert results["incomplete"]["pass"] is None
    assert results["missing"]["pass"] is None


def test_cleanup_cannot_remove_other_runs_or_applications(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    own = tmp_path / "run/artifacts"
    mount_sources = {
        "mine": own / "finalpool/ab-testing",
        "other-run": tmp_path / "other-run/artifacts/finalpool/ab-testing",
        "application": tmp_path / "services/canvas",
        "shared-auth": tmp_path / "upstream/configs/.mcp-auth",
    }
    monkeypatch.setattr(containerized.subprocess, "check_output", lambda *a, **kw: " ".join(mount_sources))
    removed = []

    def command(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if args[1] == "inspect":
            return subprocess.CompletedProcess(args, 0, json.dumps([{"Source": str(mount_sources[args[-1]])}]))
        removed.append(args[-1])
        return subprocess.CompletedProcess(args, 0, "")

    monkeypatch.setattr(containerized.subprocess, "run", command)
    containerized.cleanup_task_containers("docker", own)
    assert removed == ["mine"]


@pytest.mark.parametrize("missing", [False, True])
def test_full_run_dispatches_every_selected_task_and_preserves_errors(
    benchmark_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: bool
) -> None:
    monkeypatch.setenv("TOOLATHLON_OPENAI_BASE_URL", "http://localhost:12345/v1")
    monkeypatch.setattr(containerized, "configured_source", lambda environment: benchmark_source)
    monkeypatch.setattr(
        containerized, "check_environment", lambda *a: {"runtime": "docker", "image_id": "sha256:test"}
    )
    cleaned = []
    monkeypatch.setattr(containerized, "cleanup_task_containers", lambda runtime, artifacts: cleaned.append(artifacts))
    output = tmp_path / "run"

    def execute(args: list[str], **kwargs: Any) -> int:
        assert args[args.index("--runner") + 1] == "containerized"
        assert args[args.index("--workers") + 1] == "2"
        assert args[args.index("--timeout") + 1] == "123"
        assert kwargs["env"]["TOOLATHLON_CONTAINERIZED_MODE"] == "phased"
        config_path = Path(args[args.index("--eval_config") + 1])
        assert not config_path.is_absolute()
        assert (benchmark_source / config_path).is_file()
        selected = Path(args[args.index("--task_list") + 1]).read_text().splitlines()
        assert selected == ["ab-testing", "find-alita-paper", "k8s-mysql"]
        for name in selected[:2] if missing else selected:
            write_verdict(output / "artifacts", name, True)
        return 0

    monkeypatch.setattr(containerized, "execute", execute)
    result = containerized.run_containerized(
        environment=benchmark_source,
        output=output,
        tasks=None,
        model="provider/model",
        workers=2,
        timeout=123,
    )
    assert result["counts"]["total"] == 3
    assert result["status"] == ("error" if missing else "passed")
    assert result["pass_rate"] == (None if missing else 1.0)
    assert len(result["tasks"]) == 3
    assert cleaned == [output / "artifacts"]
    assert not list((benchmark_source / "scripts/temp_configs").iterdir())


def test_full_run_rejects_unknown_task_before_execution(
    benchmark_source: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOOLATHLON_OPENAI_BASE_URL", "http://localhost:12345/v1")
    monkeypatch.setattr(containerized, "configured_source", lambda environment: benchmark_source)
    with pytest.raises(ValueError, match="Unknown Toolathlon"):
        containerized.run_containerized(
            environment=benchmark_source,
            output=tmp_path / "run",
            tasks=["unknown"],
            model="model",
        )
    assert not (tmp_path / "run").exists()


def test_interrupted_full_run_cleans_containers_and_writes_summary(
    benchmark_source: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOOLATHLON_OPENAI_BASE_URL", "http://localhost:12345/v1")
    monkeypatch.setattr(containerized, "configured_source", lambda environment: benchmark_source)
    monkeypatch.setattr(
        containerized, "check_environment", lambda *a: {"runtime": "docker", "image_id": "sha256:test"}
    )
    cleaned = []
    monkeypatch.setattr(containerized, "cleanup_task_containers", lambda runtime, artifacts: cleaned.append(artifacts))

    def execute(*args: Any, **kwargs: Any) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(containerized, "execute", execute)
    output = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        containerized.run_containerized(environment=benchmark_source, output=output, tasks=None, model="model")
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "interrupted"
    assert result["counts"] == {"total": 3, "passed": 0, "failed": 0, "error": 3}
    assert cleaned == [output / "artifacts"]
