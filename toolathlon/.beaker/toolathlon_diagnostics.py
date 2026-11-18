"""Retain case diagnostics without depending on telemetry from the agent process."""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DIRECTORY_ENV = "TOOLATHLON_DIAGNOSTICS_DIR"
TASK_ENV = "TOOLATHLON_DIAGNOSTIC_TASK"
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_CASE_BYTES = 32 * 1024 * 1024
PART_CHARACTERS = 1024 * 1024
_SECRET = re.compile(r"token|secret|password|passwd|credential|authorization|api.?key|[_-]key", re.I)
_ASSIGNMENT = re.compile(
    r"""(?<![\w.-])(["']?[\w.-]*(?:token|secret|password|passwd|credential|authorization|api[_-]?key|[_-]key)[\w.-]*["']?\s*[:=]\s*)("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s,;}]+)""",
    re.I,
)


def redact(text: str, environment: Mapping[str, str] | None = None) -> str:
    environment = os.environ if environment is None else environment
    secrets = set()
    for name, value in sorted(environment.items(), key=lambda item: len(item[1]), reverse=True):
        if _SECRET.search(name) and len(value) >= 4 and not name.endswith(("_FILE", "_DIR", "_PATH")):
            secrets.add(value)
    try:
        configuration = json.loads(environment.get("TOOLATHLON_CONFIG_JSON", "{}"))
        # Config files may be Python or JSON. Inspect literals without executing
        # any code or opening a credential file; mask their string values too.
        for content in configuration.values():
            for node in ast.walk(ast.parse(content)):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) >= 8:
                    secrets.add(node.value)
    except (ValueError, SyntaxError, TypeError, AttributeError):
        pass
    for value in sorted(secrets, key=len, reverse=True):
        text = text.replace(value, "[REDACTED]")
    text = re.sub(
        r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----", "[REDACTED KEY]", text, flags=re.S
    )
    text = re.sub(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+", r"\1 [REDACTED]", text)
    text = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[REDACTED]@", text)
    return _ASSIGNMENT.sub(lambda match: match[1] + '"[REDACTED]"', text)


def event(message: str, *, directory: Path | None = None, environment: Mapping[str, str] | None = None) -> None:
    environment = os.environ if environment is None else environment
    text = redact(
        f"{datetime.now(UTC).isoformat()} [toolathlon {environment.get(TASK_ENV, 'setup')}] {message}",
        environment,
    )
    try:
        print(text, file=sys.stderr, flush=True)
    except OSError:
        pass
    configured = environment.get(DIRECTORY_ENV)
    directory = directory if directory is not None else Path(configured) if configured else None
    if directory is not None:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "phases.log").open("a") as stream:
                stream.write(text + "\n")
        except OSError:
            pass


@contextmanager
def phase(name: str) -> Iterator[None]:
    started = time.monotonic()
    event(f"START {name}")
    try:
        yield
    except BaseException as exc:
        event(f"FAIL {name} after {time.monotonic() - started:.2f}s: {type(exc).__name__}: {exc}")
        raise
    else:
        event(f"DONE {name} after {time.monotonic() - started:.2f}s")


def command_output(args: tuple[str, ...], returncode: int, stdout: str, stderr: str) -> None:
    configured = os.environ.get(DIRECTORY_ENV)
    if not configured:
        return
    try:
        path = Path(configured) / "docker-commands.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(redact(f"\n$ docker {json.dumps(args)}\nexit={returncode}\n{stdout}\n{stderr}\n"))
    except OSError:
        pass


def snapshot(directory: Path, *, budget: float = 6) -> None:
    """Bounded, read-only probes. Never inspect container environments or config files."""
    directory.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + budget

    def probe(args: list[str], filename: str) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        with (directory / filename).open("w") as stream:
            stream.write(f"$ {json.dumps(args)}\n")
            stream.flush()
            try:
                result = subprocess.run(args, stdout=stream, stderr=stream, timeout=min(2, remaining))
                stream.write(f"\nexit={result.returncode}\n")
            except (OSError, subprocess.TimeoutExpired) as exc:
                stream.write(f"\n{type(exc).__name__}: {exc}\n")

    info = [f"pid={os.getpid()} uid={os.getuid()} euid={os.geteuid()} python={sys.version}"]
    for filename in ("/proc/self/status", "/proc/sys/net/ipv4/ip_unprivileged_port_start"):
        try:
            text = Path(filename).read_text()
            if filename.endswith("status"):
                text = "\n".join(line for line in text.splitlines() if line.startswith(("Uid:", "Gid:", "Cap")))
            info.append(f"{filename}\n{text}")
        except OSError as exc:
            info.append(f"{filename}: {exc}")
    (directory / "identity.log").write_text("\n".join(info))
    for args, filename in (
        (["ip", "-j", "address", "show"], "interfaces.log"),
        (["ss", "-ltnp"], "listeners.log"),
        (["docker", "network", "inspect", "bridge"], "bridge.log"),
        (["docker", "info"], "docker-info.log"),
        (["df", "-h"], "disk.log"),
    ):
        probe(args, filename)


def tail(path: Path, environment: Mapping[str, str], limit: int = 4000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        return redact(stream.read().decode("utf-8", errors="replace"), environment)[-limit:]


def preprocess_error(path: Path, environment: Mapping[str, str]) -> str:
    for line in reversed(tail(path, environment, limit=65536).splitlines()):
        if line.startswith("TOOLATHLON_PREPROCESS_ERROR: "):
            return line.removeprefix("TOOLATHLON_PREPROCESS_ERROR: ")[:1000]
    return ""


def publish(trace: Any, root: Path, cache: Path, environment: Mapping[str, str]) -> None:
    # Beaker shortens TMPDIR through a symlink. Resolve these trusted roots
    # before rejecting links inside a case's diagnostic/output directories.
    root, cache = root.resolve(), cache.resolve()
    paths = [*sorted((root / "diagnostics").glob("*.log"))]
    paths += sorted((root / "run").glob("*.log"))
    paths += sorted((root / "run/diagnostics").glob("*.log"))
    paths += sorted((root / "run/artifacts").glob("*/*/*.log"))
    paths += [cache / "dockerd.log"]
    manifest = []
    remaining = MAX_CASE_BYTES
    for path in paths:
        if not path.is_file() or path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            continue
        name = path.relative_to(root).as_posix() if path.is_relative_to(root) else "outer-dockerd.log"
        size = path.stat().st_size
        keep = min(size, MAX_FILE_BYTES, remaining)
        if not keep:
            manifest.append({"file": name, "bytes": size, "retained_bytes": 0, "truncated": bool(size)})
            continue
        with path.open("rb") as stream:
            if keep < size:
                content = stream.read(keep // 2)
                stream.seek(size - (keep - keep // 2))
                content += b"\n[DIAGNOSTIC LOG TRUNCATED: middle omitted]\n" + stream.read(keep - keep // 2)
            else:
                content = stream.read(keep)
        remaining -= keep
        safe = redact(content.decode("utf-8", errors="replace"), environment)
        parts: list[str] = []
        for start in range(0, len(safe), PART_CHARACTERS):
            part = len(parts) + 1
            label = f"toolathlon.logs.{name.replace('/', '.')}.part{part}"
            trace.artifact(
                label,
                safe[start : start + PART_CHARACTERS],
                media_type="text/plain",
                filename=f"{name.replace('/', '_')}.part{part}.log",
            )
            parts.append(label)
        manifest.append(
            {"file": name, "bytes": size, "retained_bytes": keep, "truncated": keep < size, "parts": parts}
        )
    trace.artifact(
        "toolathlon.logs.manifest", manifest, media_type="application/json", filename="toolathlon-logs.json"
    )


@asynccontextmanager
async def capture(trace: Any, root: Path, cache: Path, environment: dict[str, str]) -> AsyncIterator[Path]:
    directory = root / "diagnostics"
    directory.mkdir()
    event("START case supervisor", directory=directory, environment=environment)
    failed = False
    try:
        yield directory
    except BaseException:
        failed = True
        message = redact(traceback.format_exc(), environment)
        try:
            (directory / "failure.log").write_text(message)
        except OSError:
            pass
        event(f"FAIL case supervisor: {message[-4000:]}", directory=directory, environment=environment)
        raise
    finally:
        try:
            if failed:
                await asyncio.to_thread(snapshot, directory)
            await asyncio.to_thread(publish, trace, root, cache, environment)
            event("Diagnostics attached to case trace", directory=directory, environment=environment)
        except Exception as exc:
            event(f"Diagnostic collection failed: {type(exc).__name__}: {exc}", environment=environment)
