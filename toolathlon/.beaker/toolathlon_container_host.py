"""One isolated benchmark host: its own Docker daemon, applications and official phases."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import yaml


ROOT = Path("/workspace")
CASE = Path("/case")


def run(*args: str, timeout: int = 1800) -> None:
    print("+", *args, flush=True)
    subprocess.run(args, cwd=ROOT, check=True, timeout=timeout)


def wait_docker() -> None:
    with (CASE / "dockerd.log").open("w") as log:
        # Keep the inner bridge distinct from the outer Docker bridge (172.17/16).
        # The official task uses host networking; provider aliases must reach
        # the outer bridge, rather than accidentally hitting this daemon.
        daemon = subprocess.Popen(["dockerd", "--bip", "172.30.0.1/16"], stdout=log, stderr=log)
    for _ in range(90):
        if subprocess.run(["docker", "info"], capture_output=True, timeout=5).returncode == 0:
            return
        if daemon.poll() is not None:
            break
        time.sleep(1)
    raise RuntimeError("The private Docker daemon failed: " + (CASE / "dockerd.log").read_text()[-3000:])


def cancel(_signum: int, _frame: object) -> None:
    # Let the original agent flush its case-scoped OTLP exporter before the
    # outer Docker host is removed. This daemon belongs to this case only.
    processes = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}", "--filter", "name=toolathlon-"],
        capture_output=True,
        text=True,
        timeout=2,
    )
    code = """import os, signal
for p in os.listdir('/proc'):
    if not p.isdigit() or int(p) == os.getpid():
        continue
    try:
        args = open('/proc/'+p+'/cmdline','rb').read().split(b'\\0')
        if args[0].split(b'/')[-1].startswith(b'python') and b'toolathlon_container_agent.py' in args:
            os.kill(int(p), signal.SIGTERM)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
"""
    for name in processes.stdout.splitlines():
        subprocess.run(["docker", "exec", name, "/workspace/.venv/bin/python", "-c", code], timeout=3)
    time.sleep(3)
    raise SystemExit(143)


def main() -> None:
    signal.signal(signal.SIGTERM, cancel)
    request = json.loads((CASE / "request.json").read_text())
    os.environ.pop("DOCKER_HOST", None)
    # Docker's default /etc/hosts is private to each nested container.
    for port, bridge_port in request["relay_ports"].items():
        subprocess.Popen(
            ["socat", f"TCP-LISTEN:{port},bind=127.0.0.1,reuseaddr,fork", f"TCP:host.docker.internal:{bridge_port}"]
        )
    wait_docker()
    # The inner daemon has a private image store and cannot access other cases.
    run("docker", "load", "--input", "/toolathlon-base.tar", timeout=900)
    run("tar", "-xf", "/case/source.tar", "-C", str(ROOT), timeout=120)
    for name in ("global_configs", "token_key_session"):
        target = ROOT / "configs" / f"{name}.py"
        if not target.exists():
            shutil.copyfile(ROOT / "configs" / f"{name}_example.py", target)
    # This private host has no competing instance: use the upstream default
    # ports and cluster names that the task files expect.
    (ROOT / "configs/ports_config.yaml").write_text(
        yaml.safe_dump(
            {
                "instance_prefix": "beaker-",
                "instance_suffix": "",
                "port_mappings": {},
            }
        )
    )
    if Path("/beaker-config").is_dir():
        shutil.copytree("/beaker-config", ROOT / "configs", dirs_exist_ok=True)
    if os.environ.get("TOOLATHLON_CONFIG_JSON"):
        files = json.loads(os.environ["TOOLATHLON_CONFIG_JSON"])
        if not isinstance(files, dict):
            raise ValueError("TOOLATHLON_CONFIG_JSON must map config-relative paths to text contents")
        for name, content in files.items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or not isinstance(content, str):
                raise ValueError("Invalid Toolathlon configuration file")
            target = ROOT / "configs" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
    os.environ["UV_PROJECT_ENVIRONMENT"] = str(ROOT / ".venv")
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["TOOLATHLON_CONTAINERIZED_MODE"] = "phased"
    os.environ["TOOLATHLON_PARENT_CAPTURES_RUN_LOG"] = "1"
    os.environ["TOOLATHLON_AGENT_PATCH"] = "/case/agent_harness.patch"
    run("uv", "sync", "--locked", timeout=600)
    # The original native Excel server alone needs its older MCP version.
    subprocess.run(
        ["uv", "sync", "--locked", "--only-group", "excel"],
        cwd=ROOT,
        env=os.environ | {"UV_PROJECT_ENVIRONMENT": str(ROOT / ".venv-excel")},
        check=True,
        timeout=300,
    )
    for name in ("toolathlon_container_agent.py", "toolathlon_native_trace.py"):
        shutil.copyfile(Path("/beaker") / name, ROOT / name)
    run("git", "apply", "/beaker/toolathlon_container.patch", timeout=10)
    task = request["task"]
    if request.get("system_prompt"):
        (ROOT / "tasks/finalpool" / task / "docs/agent_system_prompt.md").write_text(request["system_prompt"])
    needed = set(json.loads((ROOT / "tasks/finalpool" / task / "task_config.json").read_text())["needed_mcp_servers"])
    for service, command in (
        ("canvas", ["bash", "deployment/canvas/scripts/setup.sh", "start"]),
        ("emails", ["bash", "deployment/poste/scripts/setup.sh", "start", "false"]),
        ("woocommerce", ["bash", "deployment/woocommerce/scripts/setup.sh", "start", "81", "20"]),
        ("k8s", ["bash", "deployment/k8s/scripts/setup.sh"]),
    ):
        if service in needed or (service == "k8s" and task.startswith("k8s-")):
            run(*command)
    # Fail before running an agent if a required external-account config is absent.
    if needed & {"google_sheet", "google_calendar", "google_forms"}:
        for name in ("gcp-oauth.keys.json", "google_credentials.json"):
            if not (ROOT / "configs" / name).is_file():
                raise RuntimeError(f"Task {task} requires configs/{name}; supply TOOLATHLON_CONFIG_DIR")
    config = json.loads((ROOT / "scripts/formal_run_v0.json").read_text())
    config["global_task_config"].update(dump_path="/workspace/dumps", direct_to_dumps=True)
    for role in ("agent", "user"):
        config[role]["model"] = {"short_name": request["model"], "provider": "unified"}
    (ROOT / "beaker_eval_config.json").write_text(json.dumps(config))
    if request.get("model_params"):
        (ROOT / "beaker_model_params.json").write_text(json.dumps(request["model_params"]))
        os.environ["TOOLATHLON_MODEL_PARAMS_FILE"] = str(ROOT / "beaker_model_params.json")
    # No second scheduler: Beaker owns case admission. The official script keeps
    # the same live task container through preprocess, agent and remote grading.
    result = subprocess.run(
        [
            "bash",
            "scripts/run_single_containerized.sh",
            f"finalpool/{task}",
            "normal",
            "/case/artifacts",
            request["model"],
            "unified",
            str(request["max_steps"]),
            "beaker_eval_config.json",
            request["image"],
        ],
        cwd=ROOT,
        timeout=request["timeout"],
    )
    task_root = CASE / "artifacts/finalpool" / task
    trajectory = json.loads((task_root / "traj_log.json").read_text())
    evaluation = json.loads((task_root / "eval_res.json").read_text())
    if type(evaluation.get("pass")) is not bool or evaluation.get("failure") in (
        "evaluation_error",
        "log_file_not_found",
        "preprocess_error",
    ):
        raise RuntimeError(f"Task {task} has no valid official evaluation: {evaluation}")
    summary = {
        "runner": "containerized",
        "image": request["image"],
        "image_id": request["image_id"],
        "task": task,
        "status": "completed",
        "exit_code": result.returncode,
        "agent_status": trajectory.get("status"),
        "key_stats": trajectory.get("key_stats"),
        "evaluation": {"pass": evaluation["pass"], "details": json.dumps(evaluation, ensure_ascii=False)[:8000]},
        "workspace_files": sorted(
            p.relative_to(task_root / "workspace").as_posix()
            for p in (task_root / "workspace").rglob("*")
            if p.is_file() and not p.is_symlink()
        ),
    }
    # Keep the public trajectory format the parent already understands.
    (CASE / "artifacts/traj_log.json").write_text(json.dumps(trajectory))
    (CASE / "result.json").write_text(json.dumps(summary))


if __name__ == "__main__":
    main()
