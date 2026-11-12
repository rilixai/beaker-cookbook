"""Fetch the pinned benchmark and migrate its original agent in place."""

from __future__ import annotations

import hashlib
import json
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


MIGRATION_PATCH = Path(__file__).parent / "upstream/openai-agents-0.20.patch"


def _verify_revision(source: Path) -> None:
    revision = subprocess.check_output(
        [require_command("git"), "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise RuntimeError(f"Expected Toolathlon {UPSTREAM_REVISION}, found {revision}; use a fresh cache directory")


def verify_source(source: Path, *, require_clean: bool = True) -> None:
    """Require the exact reviewed migration; native caches allow no other edits."""
    _verify_revision(source)
    manifest = json.loads(MIGRATION_PATCH.with_suffix(".json").read_text())
    if hashlib.sha256(MIGRATION_PATCH.read_bytes()).hexdigest() != manifest["patch_sha256"]:
        raise RuntimeError("Toolathlon migration patch checksum mismatch")
    for name, expected in manifest["files"].items():
        path = source / name
        if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Toolathlon migration mismatch in {name}; rerun prepare on a clean pinned checkout")
    if require_clean:
        changed = subprocess.check_output(
            [require_command("git"), "-C", str(source), "diff", "HEAD", "--name-only"], text=True
        ).splitlines()
        if set(changed) - manifest["files"].keys():
            raise RuntimeError("The cached Toolathlon source has changes outside the SDK migration")


def migrate_source(source: Path) -> None:
    """Apply the versioned source migration once, without resetting local changes."""
    _verify_revision(source)
    try:
        verify_source(source, require_clean=False)
        return
    except RuntimeError:
        pass
    git = require_command("git")
    dirty = subprocess.check_output([git, "-C", str(source), "diff", "HEAD", "--name-only"], text=True)
    if dirty:
        raise RuntimeError("Cannot migrate a modified Toolathlon checkout; use a clean pinned checkout")
    subprocess.run([git, "-C", str(source), "apply", "--check", str(MIGRATION_PATCH)], check=True)
    subprocess.run([git, "-C", str(source), "apply", str(MIGRATION_PATCH)], check=True)
    verify_source(source)


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
    migrate_source(source)
    verify_source(source)
    env = runtime_env(source)
    env.pop("UV_FROZEN")
    subprocess.run([uv, "sync", "--locked", "--project", str(source)], env=env, check=True)
    excel_project = source / "utils/local_servers/excel_runtime"
    excel_env = env | {"UV_PROJECT_ENVIRONMENT": str(excel_project / ".venv")}
    subprocess.run([uv, "sync", "--locked", "--project", str(excel_project)], env=excel_env, check=True)
    node_dir = cache / "node"
    node_dir.mkdir(exist_ok=True)
    for name in ("package.json", "package-lock.json"):
        shutil.copyfile(Path(__file__).parent / "node" / name, node_dir / name)
    subprocess.run([npm, "ci", "--ignore-scripts", "--no-audit", "--no-fund"], cwd=node_dir, check=True)
    (cache / "sdk-migration-ready").write_text(hashlib.sha256(MIGRATION_PATCH.read_bytes()).hexdigest())
    return source


def prepared_source(cache: Path) -> Path:
    cache = native_path(cache)
    source = cache / "upstream"
    if not (source / ".venv/bin/python").is_file():
        raise RuntimeError("Run `toolathlon-baseline prepare` with this cache directory first")
    verify_source(source)
    ready = cache / "sdk-migration-ready"
    if not ready.is_file() or ready.read_text() != hashlib.sha256(MIGRATION_PATCH.read_bytes()).hexdigest():
        raise RuntimeError("SDK migration setup is incomplete; rerun `toolathlon-baseline prepare`")
    if not (source / "utils/local_servers/excel_runtime/.venv/bin/excel-mcp-server").is_file():
        raise RuntimeError("Excel MCP environment is missing; rerun `toolathlon-baseline prepare`")
    if not (cache / "node/node_modules/@modelcontextprotocol/server-filesystem/dist/index.js").is_file():
        raise RuntimeError("Filesystem MCP is missing; rerun `toolathlon-baseline prepare`")
    return source
