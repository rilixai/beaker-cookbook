"""Fetch the benchmark and keep its old SDK in a separate Python environment."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from toolathlon_baseline import UPSTREAM_REVISION, UPSTREAM_URL


def native_path(path: Path) -> Path:
    """Upstream builds shell commands from paths without quoting them."""
    path = path.resolve()
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", str(path)):
        raise ValueError("Toolathlon's native runner requires paths without spaces or shell metacharacters")
    return path


def require_command(name: str) -> str:
    command = shutil.which(name)
    if command is None:
        raise RuntimeError(f"Install {name} before running this command; see the recipe README")
    return command


def verify_source(source: Path, *, require_clean: bool = True) -> None:
    revision = subprocess.check_output(
        [require_command("git"), "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise RuntimeError(f"Expected Toolathlon {UPSTREAM_REVISION}, found {revision}; use a fresh cache directory")
    if not require_clean:
        return
    # Config examples are copied only into each run, never into the source cache.
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if dirty:
        raise RuntimeError("The cached Toolathlon source has tracked changes; use a fresh cache directory")


def runtime_env(source: Path) -> dict[str, str]:
    env = os.environ.copy()
    # Nested `uv run` calls from preprocess/evaluation must use this environment,
    # including when the CLI itself was launched from another uv project.
    env.pop("VIRTUAL_ENV", None)
    env["UV_PROJECT_ENVIRONMENT"] = str(source / ".venv")
    env["UV_FROZEN"] = "1"
    env["PATH"] = f"{source / '.venv/bin'}{os.pathsep}{env.get('PATH', '')}"
    return env


def prepare(cache: Path) -> Path:
    cache = native_path(cache)
    git = require_command("git")
    uv = require_command("uv")
    npm = require_command("npm")
    node = require_command("node")
    node_major = int(subprocess.check_output([node, "--version"], text=True).strip().lstrip("v").split(".")[0])
    if node_major < 22:
        raise RuntimeError("Toolathlon requires Node.js 22 or newer")
    cache.mkdir(parents=True, exist_ok=True)
    source = cache / "upstream"
    if not source.exists():
        # Keep incomplete network downloads out of the reusable source directory.
        staging = cache / "upstream.incomplete"
        staging.mkdir()
        try:
            subprocess.run([git, "init", str(staging)], check=True)
            subprocess.run([git, "-C", str(staging), "remote", "add", "origin", UPSTREAM_URL], check=True)
            subprocess.run([git, "-C", str(staging), "fetch", "--depth=1", "origin", UPSTREAM_REVISION], check=True)
            subprocess.run([git, "-C", str(staging), "checkout", "--detach", "FETCH_HEAD"], check=True)
            staging.rename(source)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    verify_source(source)
    env = runtime_env(source)
    env.pop("UV_FROZEN")
    subprocess.run([uv, "sync", "--locked", "--project", str(source)], env=env, check=True)
    node_dir = cache / "node"
    node_dir.mkdir(exist_ok=True)
    for name in ("package.json", "package-lock.json"):
        shutil.copyfile(Path(__file__).parent / "node" / name, node_dir / name)
    subprocess.run([npm, "ci", "--ignore-scripts", "--no-audit", "--no-fund"], cwd=node_dir, check=True)
    return source


def prepared_source(cache: Path) -> Path:
    cache = native_path(cache)
    source = cache / "upstream"
    if not (source / ".venv/bin/python").is_file():
        raise RuntimeError("Run `toolathlon-baseline prepare` with this cache directory first")
    verify_source(source)
    if not (cache / "node/node_modules/@modelcontextprotocol/server-filesystem/dist/index.js").is_file():
        raise RuntimeError("Filesystem MCP is missing; rerun `toolathlon-baseline prepare`")
    return source
