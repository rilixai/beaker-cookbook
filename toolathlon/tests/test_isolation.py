from __future__ import annotations

import json
from pathlib import Path

from toolathlon_baseline.runner import stage_task


def test_run_changes_never_modify_cached_source(tmp_path: Path) -> None:
    # A minimal source tree checks the filesystem boundary, not task quality.
    source = tmp_path / "cached-source"
    for directory in ("utils", "scripts", "configs/mcp_servers", "tasks/finalpool/find-alita-paper/docs"):
        (source / directory).mkdir(parents=True)
    for name in ("main.py", "pyproject.toml", "uv.lock"):
        (source / name).write_text("")
    for name in ("global_configs", "token_key_session"):
        (source / f"configs/{name}_example.py").write_text("# config example\n")
    task = source / "tasks/finalpool/find-alita-paper"
    (task / "task_config.json").write_text(json.dumps({"needed_mcp_servers": []}))
    (task / "docs/agent_system_prompt.md").write_text("original instructions")
    override = tmp_path / "prompt.md"
    override.write_text("custom instructions")
    run_a, run_b = tmp_path / "a", tmp_path / "b"
    run_a.mkdir()
    run_b.mkdir()
    checkout_a = stage_task(source, tmp_path, run_a, "find-alita-paper", override)
    checkout_b = stage_task(source, tmp_path, run_b, "find-alita-paper", None)
    relative = "tasks/finalpool/find-alita-paper/docs/agent_system_prompt.md"
    assert (checkout_a / relative).read_text() == "custom instructions"
    (checkout_a / relative).write_text("modified during a run")
    assert (checkout_b / relative).read_text() == "original instructions"
    assert (source / relative).read_text() == "original instructions"
