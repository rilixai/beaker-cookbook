"""Prepare the vendored Toolathlon checkout, the Docker daemon, and the task apps in a sandbox."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import time
import urllib.request
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
UPSTREAM = PROJECT / "upstream"
TASKS = UPSTREAM / "tasks" / "finalpool"
TASK_IMAGE = "lockon0927/toolathlon-task-image:1016beta"
STATE = Path(tempfile.gettempdir()) / "beaker-toolathlon"
BIN = STATE / "bin"
ASSETS = json.loads((Path(__file__).with_name("upstream_assets.json")).read_text())

KIND_URL = "https://kind.sigs.k8s.io/dl/v0.29.0/kind-linux-amd64"
KUBECTL_URL = "https://dl.k8s.io/release/v1.33.1/bin/linux/amd64/kubectl"

# Self-hosted apps started with Docker, keyed by the MCP server that needs them.
APP_COMMANDS: dict[str, tuple[str, ...]] = {
    "emails": ("deployment/poste/scripts/setup.sh", "start", "true"),
    "canvas": ("deployment/canvas/scripts/setup.sh",),
    "woocommerce": ("deployment/woocommerce/scripts/setup.sh", "start", "81", "20"),
    "k8s": ("deployment/k8s/scripts/setup.sh", "start"),
}

# Platform secrets each external service reads. Values are written to the
# untracked Toolathlon config files inside the sandbox, never to the repository.
_GOOGLE_OAUTH = ("TOOLATHLON_GOOGLE_OAUTH_CREDENTIALS_JSON", "TOOLATHLON_GCP_OAUTH_KEYS_JSON")
SERVICE_ENV: dict[str, tuple[str, ...]] = {
    "web_search": ("TOOLATHLON_SERPER_API_KEY",),
    "github": ("TOOLATHLON_GITHUB_TOKEN",),
    "huggingface": ("TOOLATHLON_HUGGINGFACE_TOKEN",),
    "wandb": ("TOOLATHLON_WANDB_API_KEY",),
    "notion": (
        "TOOLATHLON_NOTION_INTEGRATION_KEY",
        "TOOLATHLON_NOTION_INTEGRATION_KEY_EVAL",
        "TOOLATHLON_NOTION_SOURCE_PAGE_URL",
        "TOOLATHLON_NOTION_EVAL_PAGE_URL",
    ),
    "snowflake": (
        "TOOLATHLON_SNOWFLAKE_ACCOUNT",
        "TOOLATHLON_SNOWFLAKE_USER",
        "TOOLATHLON_SNOWFLAKE_ROLE",
        "TOOLATHLON_SNOWFLAKE_PRIVATE_KEY",
    ),
    "google-cloud": ("TOOLATHLON_GCP_PROJECT_ID", "TOOLATHLON_GCP_SERVICE_ACCOUNT_JSON"),
    "google_sheet": _GOOGLE_OAUTH,
    "google_forms": _GOOGLE_OAUTH,
    "google_calendar": _GOOGLE_OAUTH,
    "google_map": ("TOOLATHLON_GOOGLE_CLOUD_CONSOLE_API_KEY",),
    "youtube": ("TOOLATHLON_GOOGLE_CLOUD_CONSOLE_API_KEY",),
}

_TOKEN_FIELDS = {
    "serper_api_key": "TOOLATHLON_SERPER_API_KEY",
    "google_cloud_console_api_key": "TOOLATHLON_GOOGLE_CLOUD_CONSOLE_API_KEY",
    "gcp_project_id": "TOOLATHLON_GCP_PROJECT_ID",
    "github_token": "TOOLATHLON_GITHUB_TOKEN",
    "huggingface_token": "TOOLATHLON_HUGGINGFACE_TOKEN",
    "wandb_api_key": "TOOLATHLON_WANDB_API_KEY",
    "notion_integration_key": "TOOLATHLON_NOTION_INTEGRATION_KEY",
    "notion_integration_key_eval": "TOOLATHLON_NOTION_INTEGRATION_KEY_EVAL",
    "source_notion_page_url": "TOOLATHLON_NOTION_SOURCE_PAGE_URL",
    "eval_notion_page_url": "TOOLATHLON_NOTION_EVAL_PAGE_URL",
    "snowflake_account": "TOOLATHLON_SNOWFLAKE_ACCOUNT",
    "snowflake_user": "TOOLATHLON_SNOWFLAKE_USER",
    "snowflake_role": "TOOLATHLON_SNOWFLAKE_ROLE",
    "snowflake_warehouse": "TOOLATHLON_SNOWFLAKE_WAREHOUSE",
}
_SECRET_FILES = {
    "configs/google_credentials.json": "TOOLATHLON_GOOGLE_OAUTH_CREDENTIALS_JSON",
    "configs/gcp-oauth.keys.json": "TOOLATHLON_GCP_OAUTH_KEYS_JSON",
    "configs/gcp-service_account.keys.json": "TOOLATHLON_GCP_SERVICE_ACCOUNT_JSON",
    "configs/snowflake_rsa_key.p8": "TOOLATHLON_SNOWFLAKE_PRIVATE_KEY",
}


def task_config(task: str) -> dict[str, list[str]]:
    data = json.loads((TASKS / task / "task_config.json").read_text())
    return {
        "servers": list(data.get("needed_mcp_servers", [])),
        "local_tools": list(data.get("needed_local_tools", [])),
    }


def required_env(task: str) -> tuple[str, ...]:
    config = task_config(task)
    names: dict[str, None] = {}
    for service in (*config["servers"], *config["local_tools"]):
        names.update(dict.fromkeys(SERVICE_ENV.get(service, ())))
    return tuple(names)


def required_apps(task: str) -> tuple[str, ...]:
    return tuple(server for server in task_config(task)["servers"] if server in APP_COMMANDS)


@contextmanager
def _locked(name: str) -> Iterator[None]:
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / f"{name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _run(command: list[str], *, log: Path, env: Mapping[str, str] | None = None, timeout: float) -> None:
    with log.open("ab") as output:
        subprocess.run(
            command, cwd=UPSTREAM, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=timeout, check=True
        )


def tool_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = f"{BIN}{os.pathsep}{env.get('PATH', '')}"
    return env


def _docker_ready() -> bool:
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


def _ensure_docker() -> None:
    if _docker_ready():
        return
    log = (STATE / "dockerd.log").open("ab")
    subprocess.Popen(["dockerd"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 120
    while not _docker_ready():
        if time.monotonic() > deadline:
            raise RuntimeError(f"dockerd did not start; see {STATE / 'dockerd.log'}")
        time.sleep(1)


def _download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    BIN.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial")
    with urllib.request.urlopen(url, timeout=300) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    partial.chmod(partial.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    partial.rename(destination)


def _ensure_assets(prefix: str) -> None:
    """Fetch the large upstream files under ``prefix`` that are kept out of Git, pinned by SHA-256."""
    for relative, meta in ASSETS["files"].items():
        if not relative.startswith(prefix):
            continue
        destination = UPSTREAM / relative
        if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == meta["sha256"]:
            continue
        url = f"https://raw.githubusercontent.com/{ASSETS['repository']}/{ASSETS['commit']}/{relative}"
        partial = destination.with_name(destination.name + ".partial")
        with urllib.request.urlopen(url, timeout=600) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != meta["sha256"]:
            partial.unlink()
            raise ValueError(f"Checksum mismatch for upstream Toolathlon file {relative}")
        partial.chmod(0o755 if relative.startswith("local_binary/") else 0o644)
        partial.rename(destination)


def _write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as output:
        output.write(text)


def _write_configs() -> None:
    configs = UPSTREAM / "configs"
    if not (configs / "global_configs.py").exists():
        shutil.copyfile(configs / "global_configs_example.py", configs / "global_configs.py")
    for relative, name in _SECRET_FILES.items():
        if os.environ.get(name):
            _write_private(UPSTREAM / relative, os.environ[name])
    # run_single_decoupled.sh copies the Google OAuth files into every task
    # container, so tasks without Google access still need placeholders.
    for relative in ("configs/google_credentials.json", "configs/gcp-oauth.keys.json"):
        if not (UPSTREAM / relative).exists():
            _write_private(UPSTREAM / relative, "{}")
    overrides = {field: os.environ[name] for field, name in _TOKEN_FIELDS.items() if os.environ.get(name)}
    source = (configs / "token_key_session_example.py").read_text()
    _write_private(
        configs / "token_key_session.py",
        f"{source}\nall_token_key_session.update(json.loads({json.dumps(json.dumps(overrides))}))\n",
    )


def prepare_toolathlon() -> Path:
    """Sync the harness environment, start Docker, and pull the task image once per sandbox."""
    with _locked("setup"):
        log = STATE / "setup.log"
        if not (STATE / "ready").exists():
            _write_configs()
            _ensure_assets("local_binary/")
            _run(["uv", "sync", "--frozen"], log=log, timeout=1800)
            _ensure_docker()
            _run(["docker", "pull", f"docker.io/{TASK_IMAGE}"], log=log, timeout=1800)
            (STATE / "ready").touch()
        else:
            _ensure_docker()
    return UPSTREAM


def prepare_task(task: str) -> None:
    """Fetch the task's large files and start the self-hosted apps it needs, once per sandbox."""
    with _locked(f"task-{task}"):
        _ensure_assets(f"tasks/finalpool/{task}/")
    for app in required_apps(task):
        with _locked(f"app-{app}"):
            marker = STATE / f"app-{app}.ready"
            if marker.exists():
                continue
            if app == "k8s":
                _download(KIND_URL, BIN / "kind")
                _download(KUBECTL_URL, BIN / "kubectl")
            _run(["bash", *APP_COMMANDS[app]], log=STATE / f"app-{app}.log", env=tool_env(), timeout=2700)
            marker.touch()
