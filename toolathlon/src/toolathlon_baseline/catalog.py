"""Discover tasks from the pinned benchmark instead of maintaining a task allowlist."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from toolathlon_baseline import QUICKSTART_TASKS


def task_catalog(source: Path) -> list[dict[str, Any]]:
    tasks = []
    for path in sorted((source / "tasks/finalpool").iterdir()):
        if not path.is_dir() or not (path / "task_config.json").is_file():
            continue
        if not re.fullmatch(r"[A-Za-z0-9_-]+", path.name):
            raise ValueError(f"Unsupported task directory name: {path.name}")
        config = json.loads((path / "task_config.json").read_text())
        if not (path / "docs/task.md").is_file() or not (path / "evaluation/main.py").is_file():
            raise ValueError(f"Task {path.name} is missing its instructions or evaluator")
        tasks.append(
            {
                "id": path.name,
                "mcp_servers": config.get("needed_mcp_servers", []),
                "local_tools": config.get("needed_local_tools", []),
                "native_quickstart": path.name in QUICKSTART_TASKS,
            }
        )
    if not tasks:
        raise ValueError("No benchmark tasks found")
    return tasks


def select_tasks(source: Path, names: list[str] | None) -> list[str]:
    available = [task["id"] for task in task_catalog(source)]
    if names is None:
        return available
    unknown = set(names) - set(available)
    if unknown:
        raise ValueError(f"Unknown Toolathlon tasks: {', '.join(sorted(unknown))}; use the `tasks` command")
    if not names or len(names) != len(set(names)):
        raise ValueError("Select at least one task, without duplicate task IDs")
    return names
