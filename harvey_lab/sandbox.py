"""Build (optional) + promote + trigger one harvey-lab run on rilixai's Modal sandbox.

The sibling of ``cli.py`` for the sandbox path. ``cli.py`` runs the local
laptop loop; this file pushes the source as a Modal image (via ``rilixai
push``), optionally promotes the freshly-pushed row to
``harvey-lab@production``, then queues a run via
:class:`rilixai.RilixAIClient` against the production reference.

Typical workflows:

    # Build + promote + trigger in one shot (canonical local dev flow):
    uv run harvey_lab/sandbox.py --build

    # Build + promote only, no trigger (the CI ``push-spec.yml`` flow):
    uv run harvey_lab/sandbox.py --build --no-trigger

    # Trigger only (uses whatever's currently @production):
    uv run harvey_lab/sandbox.py

Required env vars (load via .env or export):
    RILIXAI_API_BASE_URL   — API Gateway URL from the RilixaiApiStack CDK output
    RILIXAI_API_KEY        — control-plane credential
    RILIXAI_AGENT_KEY      — agent that owns this spec/scope (or pass --agent)
    Provider keys (OPENAI_API_KEY for the agent, GOOGLE_API_KEY / GEMINI_API_KEY
    for the default Gemini judge) — bound at the project level in rilixai for
    the agent + the LLM rubric judge. The task documents are fetched at run
    time from the PUBLIC ``harveyai/harvey-labs`` repo, so no dataset asset
    token is required.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from dotenv import load_dotenv
from rilixai import RilixAIClient


load_dotenv()


REPO_ROOT = Path(__file__).resolve().parents[1]
MEMBER_PYPROJECT = REPO_ROOT / "harvey_lab" / "pyproject.toml"
SPEC_TARGET = REPO_ROOT / "harvey_lab" / "optimization" / "spec.py"
SPEC_NAME = "harvey-lab"
SCOPE_KEY = "harvey-lab"
TASK_TYPE = "harvey_lab"
DEFAULT_SPEC_REFERENCE = f"{SPEC_NAME}@production"
# The migrated spec no longer self-loads data: the optimizer sources cases from
# an uploaded JSONL dataset via ``HarveyLabDataLoader``. Every run therefore
# needs a dataset reference or the server rejects it at startup. Export + upload
# one first with ``scripts/export_harvey_lab_dataset.py`` + ``rilixai dataset
# upload --name harvey-lab-dataset <jsonl-dir>``.
DEFAULT_DATASET_REFERENCE = f"{SPEC_NAME}-dataset@production"


def _short_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return "dev"
    return result.stdout.strip() or "dev"


_BUNDLE_PROVIDED_PREFIXES = ("rilixai",)
_BUNDLE_PROVIDED_NAMES = ("cookbook-common",)


def _is_bundle_provided(dep: str) -> bool:
    name = re.split(r"[<>=!~;\[ ]", dep.strip(), maxsplit=1)[0].strip().lower().replace("_", "-")
    return name.startswith(_BUNDLE_PROVIDED_PREFIXES) or name in _BUNDLE_PROVIDED_NAMES


def _member_pip_deps() -> list[str]:
    """Read the harvey_lab member's runtime deps from its pyproject.

    The rilixai build worker only sees the bundle-root pyproject when it runs
    ``pip install /spec``, so the member's deps are shoveled in via
    ``--pip-install``. ``rilixai`` (baked in) and ``cookbook-common`` (a
    workspace member installed by the bundle root) are stripped.
    """
    data = tomllib.loads(MEMBER_PYPROJECT.read_text())
    deps = data["project"]["dependencies"]
    return [d for d in deps if not _is_bundle_provided(d)]


def push_image(version: str) -> None:
    """Run ``rilixai push`` to build a new Modal image for this spec version."""
    pip_install_args: list[str] = []
    for dep in _member_pip_deps():
        pip_install_args.extend(["--pip-install", dep])
    cmd = [
        "uv",
        "run",
        "rilixai",
        "push",
        "--source-dir",
        str(REPO_ROOT),
        "--name",
        SPEC_NAME,
        "--version",
        version,
        # Exported datasets (incl. base64-embedded documents) live under
        # scripts/_datasets/; they belong in uploaded datasets, not the spec
        # image, so keep them out of the bundle.
        "--exclude",
        "scripts/_datasets/*",
        *pip_install_args,
        str(SPEC_TARGET),
    ]
    print(f"\n→ {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)


def promote_image(version: str) -> None:
    """Run ``rilixai spec promote`` to point ``@production`` at the given version."""
    cmd = ["uv", "run", "rilixai", "spec", "promote", SPEC_NAME, version]
    print(f"\n→ {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)


def trigger_run(
    *,
    client: RilixAIClient,
    spec_reference: str,
    dataset_reference: str,
    max_metric_calls: int,
) -> str:
    response = client.create_optimization_run(
        task_type=TASK_TYPE,
        spec=spec_reference,
        scope_key=SCOPE_KEY,
        dataset_ref=dataset_reference,
        config={
            # GEPA per-run knobs (consumed by rilixai's sandbox runtime).
            "max_metric_calls": max_metric_calls,
            "reflection_minibatch_size": 3,
            "reflection_model": "openai/gpt-4.1",
            "seed": 0,
            # Harvey LAB cookbook knobs (consumed by build_spec in spec.py).
            "task_model": "openai/gpt-4.1-mini-2025-04-14",
            "task_temperature": 0.0,
            "judge_model": "gemini/gemini-3.5-flash",
            "max_turns": 40,
            "max_output_tokens": 16_000,
        },
    )
    return str(response["id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--build",
        action="store_true",
        help="Push the current harvey_lab source as a new spec image, then (by default) promote it to @production.",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="With --build, push without promoting. Lets you smoke-test a new build before flipping production.",
    )
    parser.add_argument(
        "--no-trigger",
        action="store_true",
        help="Skip the optimization-run trigger after build/promote. Used by CI ``push-spec.yml``.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Spec version to push (with --build). Defaults to v<short_sha> from git.",
    )
    parser.add_argument(
        "--spec",
        default=DEFAULT_SPEC_REFERENCE,
        help=f"rilixai spec reference for the trigger. Defaults to {DEFAULT_SPEC_REFERENCE}.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_REFERENCE,
        help=(
            f"rilixai dataset reference the run sources cases from. Defaults to "
            f"{DEFAULT_DATASET_REFERENCE} (upload it first with "
            f"`rilixai dataset upload --name {SPEC_NAME}-dataset <jsonl-dir>`)."
        ),
    )
    parser.add_argument(
        "--agent",
        default=os.environ.get("RILIXAI_AGENT_KEY"),
        help="Agent key that owns this spec/scope for the trigger. Defaults to $RILIXAI_AGENT_KEY.",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=50,
        help="GEPA metric-call budget — primary cost knob. Default: 50 (smoke).",
    )
    args = parser.parse_args()

    api_key: str | None = None
    base_url: str | None = None
    if not args.no_trigger:
        api_key = os.environ.get("RILIXAI_API_KEY")
        base_url = os.environ.get("RILIXAI_API_BASE_URL")
        if not api_key or not base_url:
            print("error: RILIXAI_API_KEY and RILIXAI_API_BASE_URL must be set.", file=sys.stderr)
            return 2
        if not args.agent:
            print("error: --agent or $RILIXAI_AGENT_KEY must be set.", file=sys.stderr)
            return 2

    if args.build:
        version = args.version or f"v{_short_sha()}"
        push_image(version)
        if not args.no_promote:
            promote_image(version)
            print(f"\n{SPEC_NAME}@production now resolves to {version}.")

    if args.no_trigger:
        return 0

    assert api_key is not None and base_url is not None  # validated above
    client = RilixAIClient(base_url=base_url, api_key=api_key, agent_key=args.agent)
    run_id = trigger_run(
        client=client,
        spec_reference=args.spec,
        dataset_reference=args.dataset,
        max_metric_calls=args.max_metric_calls,
    )
    print(f"queued run: {run_id} (spec={args.spec})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
