"""Use the VM's Docker daemon, image cache and shared Toolathlon applications."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from toolathlon_diagnostics import command_output, phase
from toolathlon_environment import case_checkout, locks, needed_services, prepare_services, shared_checkout, task_locks

from toolathlon_baseline.containerized import DEFAULT_IMAGE
from toolathlon_baseline.prepare import require_command, runtime_env


HERE = Path(__file__).parent


def docker(*args: str, timeout: int = 900) -> str:
    with phase("docker " + " ".join(args[:2])):
        try:
            result = subprocess.run(
                [require_command("docker"), *args], text=True, capture_output=True, timeout=timeout
            )
        except subprocess.TimeoutExpired as exc:
            command_output(args, -1, str(exc.stdout or ""), str(exc.stderr or "") + "\nCOMMAND TIMED OUT")
            raise
        command_output(args, result.returncode, result.stdout, result.stderr)
        if result.returncode:
            raise RuntimeError(f"Docker {args[0]} failed: {result.stderr[-3000:]}")
    return result.stdout.strip()


def prepare_docker(cache: Path, source: Path) -> str:
    """Start one daemon and build one dependency image per evaluation VM."""
    os.environ["PATH"] = os.pathsep.join(
        dict.fromkeys((*os.environ.get("PATH", os.defpath).split(os.pathsep), "/usr/local/sbin", "/usr/sbin", "/sbin"))
    )
    cache.mkdir(parents=True, exist_ok=True)
    with locks(cache / "shared-locks", ["prepare-docker"]):
        try:
            docker("info", timeout=30)
        except (RuntimeError, subprocess.TimeoutExpired):
            if os.geteuid() != 0:
                raise RuntimeError("Docker is unavailable. Use a Beaker VM sandbox with dockerd.") from None
            with (cache / "dockerd.log").open("a") as log:
                process = subprocess.Popen(
                    [require_command("dockerd")], stdout=log, stderr=log, start_new_session=True
                )
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                try:
                    docker("info", timeout=30)
                    break
                except (RuntimeError, subprocess.TimeoutExpired):
                    if process.poll() is not None:
                        raise RuntimeError("dockerd failed: " + (cache / "dockerd.log").read_text()[-2000:]) from None
                    time.sleep(1)
            else:
                raise RuntimeError("Docker daemon did not become ready in 300 seconds")
        base = os.environ.get("TOOLATHLON_DOCKER_IMAGE", DEFAULT_IMAGE)
        if not re.fullmatch(r"[A-Za-z0-9_./:@+-]+", base):
            raise ValueError("Invalid TOOLATHLON_DOCKER_IMAGE")
        try:
            image_id = docker("image", "inspect", "--format", "{{.Id}}", base)
        except RuntimeError:
            docker("pull", base)
            image_id = docker("image", "inspect", "--format", "{{.Id}}", base)
        recipe = (
            f"FROM {base}\nUSER root\nWORKDIR /workspace\nCOPY pyproject.toml uv.lock ./\n"
            "RUN UV_PROJECT_ENVIRONMENT=/workspace/.venv uv sync --locked "
            "&& UV_PROJECT_ENVIRONMENT=/workspace/.venv-excel uv sync --locked --only-group excel\n"
        )
        digest = hashlib.sha256((image_id + recipe).encode())
        for name in ("pyproject.toml", "uv.lock"):
            digest.update((source / name).read_bytes())
        tag = "toolathlon-beaker-task:" + digest.hexdigest()[:16]
        try:
            docker("image", "inspect", tag)
        except RuntimeError:
            directory = cache / "docker-task-build"
            directory.mkdir(exist_ok=True)
            for name in ("pyproject.toml", "uv.lock"):
                shutil.copyfile(source / name, directory / name)
            (directory / "Dockerfile").write_text(recipe)
            docker("build", "--tag", tag, str(directory), timeout=1200)
        # Reuse upstream's static Linux tools for the host-side Kind setup.
        tools = cache / "bin"
        tools.mkdir(exist_ok=True)
        if any(not (tools / name).is_file() for name in ("kind", "kubectl", "helm")):
            container = docker("create", base, "true")
            try:
                for name in ("kind", "kubectl", "helm"):
                    docker("cp", f"{container}:/usr/local/bin/{name}", str(tools / name), timeout=60)
                    (tools / name).chmod(0o755)
            finally:
                docker("rm", container, timeout=30)
        os.environ["PATH"] = str(tools) + os.pathsep + os.environ["PATH"]
        return tag


def provider_hosts() -> list[str]:
    """Task containers share the VM network and reach its existing local proxies."""
    hosts: set[str] = set()
    for raw in Path("/etc/hosts").read_text().splitlines():
        fields = raw.split("#", 1)[0].split()
        if len(fields) > 1 and fields[0] == "127.0.0.1":
            hosts.update(f"{name}:127.0.0.1" for name in fields[1:] if "." in name and name != "localhost.localdomain")
    return sorted(hosts)


def cleanup_case(prefix: str, output: Path) -> None:
    """Stop only this execution's containers; never touch services or other cases."""
    if not re.fullmatch(r"beaker-[a-f0-9]{32}-", prefix):
        raise ValueError("Invalid case container prefix")
    identifiers = docker("ps", "-aq", "--filter", f"name={prefix}", timeout=30).split()
    for identifier in identifiers:
        try:
            mounts = json.loads(docker("inspect", "--format", "{{json .Mounts}}", identifier, timeout=10))
        except RuntimeError as exc:
            if "No such" in str(exc):
                continue  # The official EXIT trap can remove it first.
            raise
        if not any(
            mount.get("Source") and Path(mount["Source"]).resolve().is_relative_to(output.resolve())
            for mount in mounts
        ):
            continue
        code = """import os, signal
for p in os.listdir('/proc'):
    if p.isdigit() and int(p) != os.getpid():
        try:
            args = open('/proc/'+p+'/cmdline','rb').read().split(b'\\0')
            if b'toolathlon_container_agent.py' in args:
                os.kill(int(p), signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
"""
        try:
            docker("exec", identifier, "/workspace/.venv/bin/python", "-c", code, timeout=5)
        except (RuntimeError, subprocess.TimeoutExpired):
            pass
        # A cancelled agent needs time to export its final native spans.
        deadline = time.monotonic() + 6
        status = output / "telemetry_status.json"
        while status.is_file() and time.monotonic() < deadline:
            try:
                if json.loads(status.read_text()).get("omissions") != ["Agent phase has not flushed telemetry"]:
                    break
            except ValueError:
                pass
            time.sleep(0.2)
        try:
            docker("stop", "--time", "8", identifier, timeout=15)
            docker("rm", "--force", "--volumes", identifier, timeout=30)
        except RuntimeError as exc:
            if "No such" not in str(exc):
                raise


@contextmanager
def case_lease(cache: Path, source: Path, task: str, prefix: str, output: Path) -> Iterator[None]:
    resources = task_locks(source, task)
    with locks(cache / "shared-locks", resources):
        leases = cache / "case-leases"
        leases.mkdir(exist_ok=True)
        # A supervisor killed before its finally block leaves a lease. Holding
        # the same resource locks proves its former supervisor released them;
        # remove any surviving containers before preprocessing the resource again.
        for path in leases.glob("*.json"):
            try:
                previous = json.loads(path.read_text())
            except FileNotFoundError:
                continue  # An unrelated active case can finish while we scan.
            if set(previous["resources"]).intersection(resources):
                cleanup_case(previous["prefix"], Path(previous["output"]))
                path.unlink()
        path = leases / f"{prefix}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"resources": resources, "prefix": prefix, "output": str(output)}))
        temporary.replace(path)
        try:
            yield
        finally:
            cleanup_case(prefix, output)
            path.unlink()


def collect_result(output: Path, task: str, image: str, image_id: str, exit_code: int) -> dict[str, object]:
    task_root = output / "artifacts/finalpool" / task
    trajectory = json.loads((task_root / "traj_log.json").read_text())
    evaluation = json.loads((task_root / "eval_res.json").read_text())
    if evaluation.get("pass") is None and trajectory.get("status") != "success":
        evaluation = {**evaluation, "pass": False}
    if type(evaluation.get("pass")) is not bool or evaluation.get("failure") in (
        "evaluation_error",
        "log_file_not_found",
        "preprocess_error",
    ):
        raise RuntimeError(f"Task {task} has no valid official evaluation: {evaluation}")
    summary = {
        "runner": "containerized",
        "image": image,
        "image_id": image_id,
        "task": task,
        "status": "completed",
        "exit_code": exit_code,
        "agent_status": trajectory.get("status"),
        "key_stats": trajectory.get("key_stats"),
        "evaluation": {"pass": evaluation["pass"], "details": json.dumps(evaluation, ensure_ascii=False)[:8000]},
        "workspace_files": sorted(
            p.relative_to(task_root / "workspace").as_posix()
            for p in (task_root / "workspace").rglob("*")
            if p.is_file() and not p.is_symlink()
        ),
    }
    (output / "artifacts/traj_log.json").write_text(json.dumps(trajectory))
    (output / "result.json").write_text(json.dumps(summary))
    return summary


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
) -> dict[str, object]:
    if (
        not re.fullmatch(r"[A-Za-z0-9_-]+", task)
        or not (source / "tasks/finalpool" / task / "task_config.json").is_file()
    ):
        raise ValueError(f"Unknown Toolathlon task: {task}")
    output.mkdir(parents=True, exist_ok=False)
    prefix = "beaker-" + uuid.uuid4().hex + "-"
    needed = needed_services(source, task)
    shared = None

    def cancel(*_args: object) -> None:
        raise KeyboardInterrupt()

    previous = signal.signal(signal.SIGTERM, cancel)
    try:
        image = prepare_docker(cache, source)
        image_id = docker("image", "inspect", "--format", "{{.Id}}", image)
        server_api = docker("version", "--format", "{{.Server.APIVersion}}")
        api_version = "1.44" if tuple(map(int, server_api.split("."))) >= (1, 44) else server_api
        shared = shared_checkout(cache, source)
        env = runtime_env(source)
        env["PYTHONUNBUFFERED"] = "1"
        with case_lease(cache, source, task, prefix, output):
            prepare_services(shared, cache, needed, env)
            with locks(cache / "shared-locks", ["prepare-services"]):
                checkout = case_checkout(shared, source, output, task, prefix)
            config = json.loads((checkout / "scripts/formal_run_v0.json").read_text())
            config["global_task_config"].update(dump_path="/workspace/dumps", direct_to_dumps=True)
            for role in ("agent", "user"):
                config[role]["model"] = {"short_name": model, "provider": "unified"}
            (checkout / "scripts/beaker_eval_config.json").write_text(json.dumps(config))
            if system_prompt:
                shutil.copyfile(system_prompt, checkout / "tasks/finalpool" / task / "docs/agent_system_prompt.md")
            patch = output / "agent_harness.patch"
            shutil.copyfile(HERE.parent / "harness/agent.patch.md", patch)
            status = output / "telemetry_status.json"
            status.write_text(json.dumps({"omissions": ["Agent phase has not flushed telemetry"]}))
            env.update(
                {
                    "TOOLATHLON_CONTAINERIZED_MODE": "phased",
                    "TOOLATHLON_PARENT_CAPTURES_RUN_LOG": "1",
                    "TOOLATHLON_AGENT_PATCH": str(patch),
                    "TOOLATHLON_HARNESS_HELPER": str(HERE / "toolathlon_harness.py"),
                    "TOOLATHLON_DOCKER_HOSTS": json.dumps(provider_hosts()),
                    "TOOLATHLON_DOCKER_API_VERSION": api_version,
                    "TOOLATHLON_TELEMETRY_STATUS": str(status),
                }
            )
            if model_params:
                target = checkout / "beaker_model_params.json"
                shutil.copyfile(model_params, target)
                env["TOOLATHLON_MODEL_PARAMS_FILE"] = str(target)
            else:
                env.pop("TOOLATHLON_MODEL_PARAMS_FILE", None)
            with (output / "run.log").open("w") as log, phase("official preprocess / agent / evaluation"):
                result = subprocess.run(
                    [
                        "bash",
                        "scripts/run_single_containerized.sh",
                        f"finalpool/{task}",
                        "normal",
                        str(output / "artifacts"),
                        model,
                        "unified",
                        str(max_steps),
                        "scripts/beaker_eval_config.json",
                        image,
                    ],
                    cwd=checkout,
                    env=env,
                    stdout=log,
                    stderr=log,
                    timeout=timeout,
                )
            return collect_result(output, task, image, image_id, result.returncode)
    finally:
        if shared:
            for service in needed:
                path = shared / f"service-{service}.log"
                if path.is_file():
                    try:
                        shutil.copyfile(path, output / f"shared-{service}.log")
                    except OSError:
                        pass
        signal.signal(signal.SIGTERM, previous)
