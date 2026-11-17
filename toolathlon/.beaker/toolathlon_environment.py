"""Shared Toolathlon applications and per-execution working copies in one VM."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

import yaml
from toolathlon_diagnostics import event, phase


HERE = Path(__file__).parent
SERVICES = {
    "canvas": (["bash", "deployment/canvas/scripts/setup.sh", "start"], ("canvas-docker",)),
    "emails": (["bash", "deployment/poste/scripts/setup.sh", "start", "false"], ("poste",)),
    "woocommerce": (["bash", "deployment/woocommerce/scripts/setup.sh", "start", "81", "20"], ("woo-db", "woo-wp")),
    "k8s": (["bash", "deployment/k8s/scripts/setup.sh"], ("cluster1-control-plane",)),
}
PROJECT_FILES = (
    "configs",
    "deployment/k8s",
    "deployment/canvas/logs",
    "global_preparation/check_installation.py",
    "local_binary/github-mcp-server",
    "scripts",
    "utils",
    "main.py",
    "pyproject.toml",
    "uv.lock",
    "toolathlon_container_agent.py",
    "toolathlon_native_trace.py",
)


@contextmanager
def locks(directory: Path, names: list[str]) -> Iterator[None]:
    """Process-wide locks also cover retries and different candidate processes."""
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for name in sorted(set(names)):
            path = directory / (hashlib.sha256(name.encode()).hexdigest() + ".lock")
            stream = stack.enter_context(path.open("a"))
            with phase(f"wait for shared resource {name}"):
                fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def task_locks(source: Path, task: str) -> list[str]:
    result = [f"task:{task}"]
    groups = json.loads((source / "tasks/finalpool/task_conflict.json").read_text())["conflict_groups"]
    for group in groups:
        if task in group:
            result.append("conflict:" + ",".join(sorted(group)))
    # Kind setup/preprocess can delete clusters and rewrite shared kubeconfigs.
    if task.startswith("k8s-"):
        result.append("service:k8s")
    return result


def needed_services(source: Path, task: str) -> set[str]:
    config = json.loads((source / "tasks/finalpool" / task / "task_config.json").read_text())
    needed = set(config["needed_mcp_servers"]).intersection(SERVICES)
    if task.startswith("k8s-"):
        needed.add("k8s")
    return needed


def copy_entry(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", ".venv", ".mcp-auth"))
    else:
        shutil.copy2(source, destination)


def shared_checkout(cache: Path, source: Path) -> Path:
    # Configuration and trusted bootstrap changes invalidate this VM-local copy.
    digest = hashlib.sha256()
    for name in (
        "toolathlon_environment.py",
        "toolathlon_container.patch",
        "toolathlon_container_agent.py",
        "toolathlon_native_trace.py",
    ):
        digest.update((HERE / name).read_bytes())
    digest.update(os.environ.get("TOOLATHLON_CONFIG_JSON", "").encode())
    config_dir = os.environ.get("TOOLATHLON_CONFIG_DIR")
    if config_dir:
        directory = Path(config_dir).resolve(strict=True)
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                digest.update(path.relative_to(directory).as_posix().encode())
                digest.update(path.read_bytes())
    root = cache / ("shared-environment-" + digest.hexdigest()[:16])
    with locks(cache / "shared-locks", ["prepare-checkout"]):
        if (root / ".ready").is_file():
            return root
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir()
        # Copy reviewed source files, not the Git database, dependency trees or
        # mutable application data from earlier executions.
        tracked = subprocess.check_output(["git", "-C", str(source), "ls-files", "-z"]).decode().split("\0")
        for name in filter(None, tracked):
            copy_entry(source / name, root / name)
        for name in ("global_configs", "token_key_session"):
            shutil.copyfile(root / "configs" / f"{name}_example.py", root / "configs" / f"{name}.py")
        (root / "configs/ports_config.yaml").write_text(
            yaml.safe_dump({"instance_prefix": "beaker-", "instance_suffix": "", "port_mappings": {}})
        )
        if config_dir:
            shutil.copytree(config_dir, root / "configs", dirs_exist_ok=True)
        files = json.loads(os.environ.get("TOOLATHLON_CONFIG_JSON", "{}"))
        if not isinstance(files, dict):
            raise ValueError("TOOLATHLON_CONFIG_JSON must map config-relative paths to text contents")
        for name, content in files.items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or not isinstance(content, str):
                raise ValueError("Invalid Toolathlon configuration file")
            target = root / "configs" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            target.chmod(0o600)
        for name in (".venv", ".venv-excel"):
            (root / name).symlink_to(source / name, target_is_directory=True)
        subprocess.run(["git", "apply", str(HERE / "toolathlon_container.patch")], cwd=root, check=True)
        for name in ("toolathlon_container_agent.py", "toolathlon_native_trace.py"):
            shutil.copyfile(HERE / name, root / name)
        poste = root / "deployment/poste/scripts/setup.sh"
        poste.write_text(poste.read_text().replace("analogic/poste.io:2.5.5", "analogic/poste.io:2.5.24"))
        (root / ".ready").write_text("")
    return root


def prepare_services(root: Path, cache: Path, needed: set[str], env: dict[str, str]) -> None:
    """Deploy an application once. Never restart a live service under another case."""
    with locks(cache / "shared-locks", ["prepare-services"]):
        for service in sorted(needed):
            marker = root / f".service-{service}-ready"
            command, containers = SERVICES[service]
            suffix = yaml.safe_load((root / "configs/ports_config.yaml").read_text()).get("instance_suffix", "")
            containers = tuple(
                f"cluster{suffix}1-control-plane" if name == "cluster1-control-plane" else name + suffix
                for name in containers
            )
            if marker.is_file():
                for name in containers:
                    state = subprocess.check_output(
                        ["docker", "inspect", "--format", "{{.State.Running}}", name], text=True, timeout=30
                    ).strip()
                    if state != "true":
                        raise RuntimeError(f"Shared service {service} stopped; refusing to reset active cases")
                event(f"Reusing shared service {service}")
                continue
            # Detach the setup process group so Canvas's background HTTPS proxy
            # survives the case supervisor. Reap that group if setup is cancelled.
            with phase(f"prepare shared service {service}"), (root / f"service-{service}.log").open("w") as log:
                process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=log, start_new_session=True)
                try:
                    status = process.wait(timeout=1800)
                    if status:
                        raise RuntimeError(f"Shared {service} setup exited {status}; see service-{service}.log")
                    for name in containers:
                        state = subprocess.check_output(
                            ["docker", "inspect", "--format", "{{.State.Running}}", name], text=True, timeout=30
                        ).strip()
                        if state != "true":
                            raise RuntimeError(f"Shared {service} setup did not leave {name} running")
                except BaseException:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=10)
                    except ProcessLookupError:
                        pass
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
            marker.write_text("")


def case_checkout(root: Path, source: Path, output: Path, task: str, prefix: str) -> Path:
    checkout = output / "source"
    checkout.mkdir()
    for name in PROJECT_FILES:
        path = root / name
        if path.exists():
            copy_entry(path, checkout / name)
    copy_entry(root / "tasks/finalpool" / task, checkout / "tasks/finalpool" / task)
    for name in (".venv", ".venv-excel"):
        (checkout / name).symlink_to(source / name, target_is_directory=True)
    # OAuth refresh state belongs to the shared application account, not to an
    # ephemeral checkout. The official runner bind-mounts this path into the task.
    auth = root / "configs/.mcp-auth"
    auth.mkdir(exist_ok=True)
    target = checkout / "configs/.mcp-auth"
    if target.exists():
        shutil.rmtree(target)
    target.symlink_to(auth, target_is_directory=True)
    ports = yaml.safe_load((root / "configs/ports_config.yaml").read_text())
    (checkout / "configs/ports_config.yaml").write_text(yaml.safe_dump(ports | {"instance_prefix": prefix}))
    needed = set(
        json.loads((checkout / "tasks/finalpool" / task / "task_config.json").read_text())["needed_mcp_servers"]
    )
    for name in ("gcp-oauth.keys.json", "google_credentials.json"):
        path = checkout / "configs" / name
        if path.is_file():
            continue
        if needed & {"google_sheet", "google_calendar", "google_forms"}:
            raise RuntimeError(f"Task {task} requires configs/{name}; supply TOOLATHLON_CONFIG_DIR")
        path.write_text("{}")
    return checkout
