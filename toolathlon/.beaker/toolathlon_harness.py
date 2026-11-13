"""Apply candidate changes to agent code only, restoring it before official grading."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath


AGENT_DIRECTORIES = (
    "utils/api_model/",
    "utils/aux_tools/",
    "utils/conversation/",
    "utils/data_processing/",
    "utils/data_structures/",
    "utils/general/",
    "utils/logging/",
    "utils/mcp/",
    "utils/misc/",
    "utils/openai_agents_monkey_patch/",
    "utils/roles/",
    "utils/system_prompts/",
    "utils/task_runner/",
    "configs/mcp_servers/",
    "local_servers/",
)
AGENT_FILES = {"utils/__init__.py", "utils/status_manager.py", "scripts/containerized/container_agent.py"}
DEPENDENCY_FILES = {"pyproject.toml", "package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "uv.lock"}


def patch_paths(patch: Path) -> list[str]:
    """Accept textual Git diffs for agent code; never task, grading or runtime files."""
    content = patch.read_text()
    if all(not line.strip() or line.startswith("#") for line in content.splitlines()):
        return []
    paths = set()
    for line in content.splitlines():
        if not line.startswith("diff --git "):
            continue
        match = re.fullmatch(r"diff --git a/([A-Za-z0-9_./-]+) b/([A-Za-z0-9_./-]+)", line)
        if not match or match[1] != match[2]:
            raise ValueError("Agent patches must use Git diffs without renames or quoted paths")
        name = match[1]
        parts = PurePosixPath(name).parts
        if (
            name.startswith("/")
            or ".." in parts
            or PurePosixPath(name).name in DEPENDENCY_FILES
            or name.endswith(".lock")
            or any(part in {".git", ".venv", ".venv-excel", "node_modules", "__pycache__"} for part in parts)
            or not (name in AGENT_FILES or name.startswith(AGENT_DIRECTORIES))
        ):
            raise ValueError(f"Outside the editable agent harness: {name}")
        paths.add(name)
    if re.search(r"^(?:rename|copy) (?:from|to) ", content, re.MULTILINE):
        raise ValueError("Agent patches may not rename or copy paths")
    if not paths or re.search(r"^(?:old|new|deleted)(?: file)? mode (?!100644\b|100755\b)", content, re.MULTILINE):
        raise ValueError("Agent patches may change regular text files only")
    # From the cookbook's toolathlon/ subdirectory, Git otherwise filters out
    # upstream-relative paths as if they belonged outside that subdirectory.
    with tempfile.TemporaryDirectory(prefix="toolathlon-patch-syntax-") as directory:
        stat = subprocess.check_output(
            ["git", "apply", "--numstat", "-z", str(patch.resolve())], cwd=directory, text=True
        )
    records = [entry.split("\t", 2) for entry in stat.split("\0") if entry]
    if any(len(entry) != 3 or not entry[0].isdigit() or not entry[1].isdigit() for entry in records):
        raise ValueError("Agent patches may not contain binary files")
    if {entry[2] for entry in records} != paths:
        raise ValueError("Patch paths do not match its Git headers")
    return sorted(paths)


def command(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=60)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Agent harness {args[1]} failed: {exc.output[-3000:]}") from exc


# These run in the task container using its original interpreter, before imports
# of candidate modules. The backup and manifest stay outside that container.
_INSPECT = """
import json, pathlib, sys
result = {}
for name in json.loads(sys.argv[1]):
    path = pathlib.Path('/workspace') / name
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Harness path must not traverse symlinks: ' + name)
    if path.exists() and not path.is_file():
        raise ValueError('Harness patch target must be a regular file: ' + name)
    result[name] = path.exists()
print(json.dumps(result))
"""
_REMOVE = """
import importlib.util, json, pathlib, shutil, sys
for name in json.loads(sys.argv[1]):
    path = pathlib.Path('/workspace') / name
    if any(p.is_symlink() for p in path.parents):
        raise ValueError('Cannot restore harness through a symlink: ' + name)
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    if path.suffix == '.py':
        pathlib.Path(importlib.util.cache_from_source(str(path))).unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
"""


def stage(runtime: str, container: str, patch: Path, backup: Path) -> None:
    paths = patch_paths(patch)
    backup.mkdir(parents=True, exist_ok=False)
    manifest = (
        json.loads(
            command(runtime, "exec", container, "/workspace/.venv/bin/python", "-c", _INSPECT, json.dumps(paths))
        )
        if paths
        else {}
    )
    for name, existed in manifest.items():
        if existed:
            target = backup / name
            target.parent.mkdir(parents=True, exist_ok=True)
            command(runtime, "cp", f"{container}:/workspace/{name}", str(target))
    (backup / "manifest.json").write_text(json.dumps(manifest))
    if not paths:
        return
    remote_patch = "/run/beaker-agent.patch"
    command(runtime, "cp", str(patch), f"{container}:{remote_patch}")
    command(runtime, "exec", container, "git", "-C", "/workspace", "apply", "--check", remote_patch)
    command(runtime, "exec", container, "git", "-C", "/workspace", "apply", remote_patch)
    command(runtime, "exec", container, "rm", "-f", remote_patch)


def restore(runtime: str, container: str, backup: Path) -> None:
    manifest = json.loads((backup / "manifest.json").read_text())
    if not manifest:
        return
    command(runtime, "exec", container, "/workspace/.venv/bin/python", "-c", _REMOVE, json.dumps(list(manifest)))
    for name, existed in manifest.items():
        if existed:
            command(runtime, "cp", str(backup / name), f"{container}:/workspace/{name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("stage", "restore"))
    parser.add_argument("runtime")
    parser.add_argument("container")
    parser.add_argument("backup", type=Path)
    parser.add_argument("--patch", type=Path)
    args = parser.parse_args()
    if args.phase == "stage":
        if args.patch is None:
            parser.error("stage requires --patch")
        stage(args.runtime, args.container, args.patch, args.backup)
    else:
        restore(args.runtime, args.container, args.backup)
