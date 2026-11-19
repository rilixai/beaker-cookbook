"""Use local Docker, or own a Docker daemon inside a Beaker Linux VM worker."""

from __future__ import annotations

import asyncio
import hashlib
import os
import platform
import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import requests


COMPOSE_VERSION = "v2.39.4"
COMPOSE_SHA256 = {
    "x86_64": "7af95166a730b87e172d4fc9aefea8725d3c6c7327d59149267b452114ddb7d4",
    "aarch64": "49082844b87f03cdcd5f5bbef1ba8c9c897b7a2dfb80cea18d61ec8ca6117e0c",
}


async def command_ok(*args: str) -> bool:
    process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    try:
        return await asyncio.wait_for(process.wait(), 10) == 0
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise


def install_compose(config_dir: Path) -> None:
    arch = platform.machine()
    expected = COMPOSE_SHA256.get(arch)
    if expected is None:
        raise RuntimeError(f"Unsupported Docker Compose architecture: {arch}")
    url = f"https://github.com/docker/compose/releases/download/{COMPOSE_VERSION}/docker-compose-linux-{arch}"
    plugin = config_dir / "cli-plugins" / "docker-compose"
    plugin.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=(30, 120)) as response:
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != expected:
            raise ValueError("Docker Compose checksum mismatch")
        plugin.write_bytes(response.content)
    plugin.chmod(0o755)


@asynccontextmanager
async def docker_runtime(scratch_dir: Path) -> AsyncIterator[None]:
    del scratch_dir
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is missing. Install Docker Desktop locally; hosted builds install docker.io.")
    if await command_ok("docker", "info"):
        if not await command_ok("docker", "compose", "version"):
            raise RuntimeError("Docker Compose v2 is required")
        yield
        return
    if platform.system() != "Linux" or os.geteuid() != 0 or shutil.which("dockerd") is None:
        raise RuntimeError("Start Docker locally, or use a Beaker Linux VM worker with docker.io installed")

    # Each rollout process owns its daemon and its temporary data. No task gets
    # the socket, provider keys, or another task's filesystem mounted into it.
    # Keep Unix socket and containerd paths below Linux's socket path limit.
    work = Path(tempfile.mkdtemp(prefix="tb-docker-", dir="/tmp"))
    config_dir = work / "config"
    previous = {key: os.environ.get(key) for key in ("DOCKER_HOST", "DOCKER_CONFIG")}
    socket = f"unix://{work / 'docker.sock'}"
    process: asyncio.subprocess.Process | None = None
    try:
        await asyncio.to_thread(install_compose, config_dir)
        os.environ.update(DOCKER_HOST=socket, DOCKER_CONFIG=str(config_dir))
        with (work / "daemon.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                "dockerd",
                "--host",
                socket,
                "--data-root",
                str(work / "data"),
                "--exec-root",
                str(work / "exec"),
                "--pidfile",
                str(work / "daemon.pid"),
                "--bip",
                "10.249.0.1/24",
                "--default-address-pool",
                "base=10.250.0.0/16,size=24",
                stdout=log,
                stderr=log,
            )
        async with asyncio.timeout(90):
            while not await command_ok("docker", "info"):
                if process.returncode is not None:
                    detail = (work / "daemon.log").read_text(errors="replace")[-4000:]
                    raise RuntimeError(f"Docker daemon failed to start: {detail}")
                await asyncio.sleep(0.5)
        yield
    finally:
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 20)
            except TimeoutError:
                process.kill()
                await process.wait()
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(work, ignore_errors=True)
