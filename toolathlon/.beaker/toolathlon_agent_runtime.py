"""Create the locked runtime for the native OpenAI Agents implementation."""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import subprocess
from pathlib import Path


def prepare_agent_runtime(cache: Path, source: Path) -> Path:
    manifest = Path(__file__).with_name("agent_runtime")
    names = ("pyproject.toml", "uv.lock")
    digest = hashlib.sha256(b"".join((manifest / name).read_bytes() for name in names)).hexdigest()[:16]
    root = cache / f"agent-runtime-{digest}"
    root.mkdir(exist_ok=True)
    with (root / "setup.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        python = root / ".venv/bin/python"
        marker = root / "ready"
        if marker.is_file() and python.is_file():
            return root / ".venv"
        for name in names:
            shutil.copyfile(manifest / name, root / name)
        env = os.environ.copy()
        for name in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "UV_FROZEN", "UV_NO_SYNC"):
            env.pop(name, None)
        subprocess.run(
            [
                "uv",
                "sync",
                "--locked",
                "--no-dev",
                "--project",
                str(root),
                "--python",
                str(source / ".venv/bin/python"),
            ],
            check=True,
            env=env,
        )
        marker.write_text(digest)
        return root / ".venv"
