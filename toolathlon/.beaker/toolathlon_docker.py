"""Docker backend for Beaker: each case owns a private benchmark host and daemon."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import tarfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from toolathlon_baseline.containerized import DEFAULT_IMAGE
from toolathlon_baseline.prepare import require_command


HERE = Path(__file__).parent
TRUST = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "NODE_EXTRA_CA_CERTS")
FORWARDED = (
    "TOOLATHLON_OPENAI_BASE_URL",
    "TOOLATHLON_OPENAI_API_KEY",
    "TOOLATHLON_OTLP_ENDPOINT",
    "TOOLATHLON_OTLP_TOKEN",
    "TOOLATHLON_CONFIG_JSON",
)


def docker(*args: str, timeout: int = 900) -> str:
    result = subprocess.run([require_command("docker"), *args], text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Docker {args[0]} failed: {result.stderr[-3000:]}")
    return result.stdout.strip()


def prepare_docker(cache: Path) -> str:
    """Prepare immutable tools once per VM; never share live service state."""
    # Beaker gives evaluator children a minimal PATH without sbin. Debian's
    # docker.io installs dockerd/docker-proxy there; dockerd also needs iptables.
    os.environ["PATH"] = os.pathsep.join(
        dict.fromkeys((*os.environ.get("PATH", os.defpath).split(os.pathsep), "/usr/local/sbin", "/usr/sbin", "/sbin"))
    )
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "docker-setup.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            docker("info", timeout=10)
        except RuntimeError:
            if os.geteuid() != 0:
                raise RuntimeError(
                    "Docker is unavailable. Start Docker or use a Beaker VM sandbox with dockerd."
                ) from None
            with (cache / "dockerd.log").open("a") as log:
                process = subprocess.Popen(
                    [require_command("dockerd")], stdout=log, stderr=log, start_new_session=True
                )
            for _ in range(90):
                try:
                    docker("info", timeout=5)
                    break
                except RuntimeError:
                    if process.poll() is not None:
                        raise RuntimeError(
                            "dockerd failed; the containerized runner needs a VM sandbox: "
                            + (cache / "dockerd.log").read_text()[-2000:]
                        ) from None
                    time.sleep(1)
            else:
                raise RuntimeError("Docker daemon did not become ready in 90 seconds")
        base = os.environ.get("TOOLATHLON_DOCKER_IMAGE", DEFAULT_IMAGE)
        if not re.fullmatch(r"[A-Za-z0-9_./:@+-]+", base):
            raise ValueError("Invalid TOOLATHLON_DOCKER_IMAGE")
        try:
            image_id = docker("image", "inspect", "--format", "{{.Id}}", base)
        except RuntimeError:
            docker("pull", base)
            image_id = docker("image", "inspect", "--format", "{{.Id}}", base)
        archive = cache / ("docker-base-" + image_id.removeprefix("sha256:")[:16] + ".tar")
        if not archive.exists():
            temporary = archive.with_suffix(".incomplete")
            docker("image", "save", "--output", str(temporary), base, timeout=900)
            temporary.replace(archive)
        # Only this small host layer is new; the task still uses the original image.
        recipe = (
            f"FROM {base}\nUSER root\nRUN apt-get update && apt-get install -y "
            "docker.io socat iproute2 psmisc netcat-openbsd && rm -rf /var/lib/apt/lists/*\nVOLUME /var/lib/docker\n"
        )
        tag = "toolathlon-beaker-host:" + hashlib.sha256((image_id + recipe).encode()).hexdigest()[:16]
        try:
            docker("image", "inspect", tag)
        except RuntimeError:
            directory = cache / "docker-host-build"
            directory.mkdir(exist_ok=True)
            (directory / "Dockerfile").write_text(recipe)
            docker("build", "--tag", tag, str(directory), timeout=1200)
        return tag


@contextmanager
def network(output: Path) -> Iterator[tuple[list[str], dict[int, int]]]:
    """Relay this case's loopback services through free, unprivileged bridge ports."""
    gateway = docker("network", "inspect", "bridge", "--format", "{{(index .IPAM.Config 0).Gateway}}")
    hosts, ports = [f"host.docker.internal:{gateway}"], set()
    for raw in Path("/etc/hosts").read_text().splitlines():
        fields = raw.split("#", 1)[0].split()
        if len(fields) > 1 and fields[0] == "127.0.0.1":
            for name in fields[1:]:
                if "." in name and name != "localhost.localdomain":
                    # The official task uses its private host's network namespace.
                    # Its loopback listener maps the unchanged provider port to
                    # this case's free port on the outer Docker bridge.
                    hosts.append(f"{name}:127.0.0.1")
                    ports.add(443)
    for name in ("TOOLATHLON_OPENAI_BASE_URL", "TOOLATHLON_OTLP_ENDPOINT"):
        url = urlsplit(os.environ.get(name, ""))
        if url.hostname in ("127.0.0.1", "localhost", "::1"):
            ports.add(url.port or (443 if url.scheme == "https" else 80))
    processes = []
    mappings = {}
    try:
        for port in sorted(ports):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
                reservation.bind((gateway, 0))
                bridge_port = reservation.getsockname()[1]
            log_path = output / f"relay-{port}.log"
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    [
                        require_command("socat"),
                        f"TCP4-LISTEN:{bridge_port},bind={gateway},reuseaddr,fork",
                        f"TCP4:127.0.0.1:{port}",
                    ],
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            processes.append(process)
            for _ in range(30):
                if process.poll() is not None:
                    detail = log_path.read_text(errors="replace")[-2000:]
                    raise RuntimeError(
                        f"Cannot expose case proxy port {port} through {gateway}:{bridge_port} "
                        f"(socat exit {process.returncode}): {detail}"
                    )
                try:
                    with socket.create_connection((gateway, bridge_port), timeout=1):
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                detail = log_path.read_text(errors="replace")[-2000:]
                raise RuntimeError(f"Docker bridge proxy {port} did not start on {gateway}:{bridge_port}: {detail}")
            mappings[port] = bridge_port
        yield hosts, mappings
    finally:
        # Each case owns its relays, including socat's connection-handler children.
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=1)


def run_containerized_agent(
    *,
    cache: Path,
    source: Path,
    output: Path,
    task: str,
    model: str,
    max_steps: int,
    timeout: int,
    system_prompt: Path | None,
    model_params: Path | None,
) -> dict:
    if (
        not re.fullmatch(r"[A-Za-z0-9_-]+", task)
        or not (source / "tasks/finalpool" / task / "task_config.json").is_file()
    ):
        raise ValueError(f"Unknown Toolathlon task: {task}")
    image = prepare_docker(cache)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(HERE.parent / "harness/agent.patch.md", output / "agent_harness.patch")
    with network(output) as (hosts, ports):
        base = os.environ.get("TOOLATHLON_DOCKER_IMAGE", DEFAULT_IMAGE)
        image_id = docker("image", "inspect", "--format", "{{.Id}}", base)
        archive = cache / ("docker-base-" + image_id.removeprefix("sha256:")[:16] + ".tar")
        request = {
            "task": task,
            "model": model,
            "max_steps": max_steps,
            "timeout": timeout,
            "image": base,
            "image_id": image_id,
            "relay_ports": ports,
            "system_prompt": system_prompt.read_text() if system_prompt else None,
            "model_params": json.loads(model_params.read_text()) if model_params else None,
        }
        (output / "request.json").write_text(json.dumps(request))
        with tarfile.open(output / "source.tar", "w") as source_archive:

            def select(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
                parts = Path(member.name).parts
                if any(p in {"node_modules", "__pycache__", ".venv", ".venv-excel", ".cache", ".env"} for p in parts):
                    return None
                return member

            source_archive.add(source, arcname=".", filter=select)
        env = os.environ.copy()
        env["TOOLATHLON_DOCKER_HOSTS"] = json.dumps(hosts)
        env["TOOLATHLON_TELEMETRY_STATUS"] = "/case/telemetry_status.json"
        (output / "telemetry_status.json").write_text(
            json.dumps({"omissions": ["Agent phase has not flushed telemetry"]})
        )
        bootstrap = output / "bootstrap"
        bootstrap.mkdir()
        for filename in (
            "toolathlon_container_host.py",
            "toolathlon_container_agent.py",
            "toolathlon_native_trace.py",
            "toolathlon_container.patch",
            "toolathlon_harness.py",
        ):
            shutil.copyfile(HERE / filename, bootstrap / filename)
        args = [
            "run",
            "--name",
            "toolathlon-beaker-" + uuid.uuid4().hex,
            "--privileged",
            "--cgroupns",
            "host",
            "-v",
            f"{archive}:/toolathlon-base.tar:ro",
            "--add-host",
            hosts[0],
            "-v",
            f"{output}:/case",
            "-v",
            f"{bootstrap}:/beaker:ro",
        ]
        name = args[2]
        mounted: set[str] = set()
        for key in TRUST:
            if env.get(key) and Path(env[key]).is_file():
                original = Path(env[key]).resolve()
                destination = "/beaker-ca/" + hashlib.sha256(str(original).encode()).hexdigest()[:16] + ".pem"
                if destination not in mounted:
                    args += ["-v", f"{original}:{destination}:ro"]
                    mounted.add(destination)
                args += ["-e", key]
                env[key] = destination
        if env.get("SSL_CERT_DIR"):
            empty = output / "empty-ca"
            empty.mkdir()
            env["SSL_CERT_DIR"] = "/case/empty-ca"
            args += ["-e", "SSL_CERT_DIR"]
        config_dir = env.get("TOOLATHLON_CONFIG_DIR")
        if config_dir:
            directory = Path(config_dir).resolve(strict=True)
            if not directory.is_dir():
                raise ValueError("TOOLATHLON_CONFIG_DIR must be a directory")
            args += ["-v", f"{directory}:/beaker-config:ro"]
        for key in (*FORWARDED, "TOOLATHLON_DOCKER_HOSTS", "TOOLATHLON_TELEMETRY_STATUS"):
            if key in env:
                args += ["-e", key]
        args += [image, "/workspace/.venv/bin/python", "/beaker/toolathlon_container_host.py"]

        # SIGTERM must tear down the private Docker host, not just its CLI client.
        def cancel(*_args: object) -> None:
            raise KeyboardInterrupt()

        previous = signal.signal(signal.SIGTERM, cancel)
        try:
            with (output / "run.log").open("w") as log:
                result = subprocess.run(["docker", *args], env=env, stdout=log, stderr=log, timeout=timeout + 1800)
            if result.returncode or not (output / "result.json").is_file():
                raise RuntimeError(
                    f"Containerized task {task} could not complete: "
                    + (output / "run.log").read_text(errors="replace")[-4000:]
                )
            return json.loads((output / "result.json").read_text())
        finally:
            try:
                docker("stop", "--time", "8", name, timeout=10)
            except (RuntimeError, subprocess.TimeoutExpired):
                pass
            try:
                docker("rm", "--force", "--volumes", name, timeout=10)
            except (RuntimeError, subprocess.TimeoutExpired):
                pass
            signal.signal(signal.SIGTERM, previous)
