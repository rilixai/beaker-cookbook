"""Build (optional) + promote + trigger one apex-agents run on rilixai's Modal sandbox.

The sibling of ``cli.py`` for the sandbox path. ``cli.py`` runs the
local laptop loop; this file pushes the source as a Modal image (via
``rilixai push``), optionally promotes the freshly-pushed row to
``apex-agents@production``, then queues a run via
:class:`rilixai.RilixAIClient` against the production reference.

Version strategy (mirrors rilix/rilix PR #1381):

* No version pinned in ``@spec(...)``. ``--build`` defaults the push
  version to ``v<short_sha>`` so each ``main`` push lands an
  immutable SpecVersion automatically.
* After push, ``--build`` also calls ``rilixai spec promote`` to
  point ``apex-agents@production`` at the just-pushed version. Pass
  ``--no-promote`` to push without promoting (smoke / regression).
* The trigger defaults to ``apex-agents@production``, which rilixai
  resolves server-side to the currently promoted version. Override
  ``--spec apex-agents@v<sha>`` to pin a specific build.

Typical workflows:

    # Build + promote + trigger in one shot (canonical local dev flow):
    uv run apex_agents/sandbox.py --build

    # Build + promote only, no trigger (the CI ``push-spec.yml`` flow —
    # ships the image and flips @production without spending LLM tokens):
    uv run apex_agents/sandbox.py --build --no-trigger

    # Trigger only (uses whatever's currently @production):
    uv run apex_agents/sandbox.py

    # Pin a specific version (smoke / regression):
    uv run apex_agents/sandbox.py --spec apex-agents@v1a2b3c4

    # Push without promoting (smoke-test a new build before flipping prod):
    uv run apex_agents/sandbox.py --build --no-promote --spec apex-agents@v1a2b3c4

Required env vars (load via .env or export):
    RILIXAI_API_BASE_URL   — API Gateway URL from the RilixaiApiStack CDK output
    RILIXAI_API_KEY        — control-plane credential
    HUGGING_FACE_HUB_TOKEN — bound at the project level in rilixai;
                             needed inside the sandbox to download the
                             private ``mercor/apex-agents`` dataset.
    Provider keys (OPENAI_API_KEY for the agent + reflection LM,
    GOOGLE_API_KEY for the default Gemini judge — GEMINI_API_KEY also
    works as a fallback) — also bound at the project level for the
    agent + the LLM rubric judge.
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


# Cookbook layout the script is hardwired against. This file lives at
# ``apex_agents/sandbox.py``; parents[1] is the cookbook repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
MEMBER_PYPROJECT = REPO_ROOT / "apex_agents" / "pyproject.toml"
SPEC_TARGET = REPO_ROOT / "apex_agents" / "optimization" / "spec.py"
SPEC_NAME = "apex-agents"
SCOPE_KEY = "apex-agents"
TASK_TYPE = "apex_agent"
DEFAULT_SPEC_REFERENCE = f"{SPEC_NAME}@production"
# The migrated spec no longer self-loads data: the optimizer sources cases from
# an uploaded JSONL dataset via ``ApexAgentsDataLoader``. Every run therefore
# needs a dataset reference or the server rejects it at startup ("Optimization
# run requires a dataset input artifact"). Upload one first with
# ``rilixai dataset upload --name apex-agents-dataset <jsonl-dir>``; this is the
# reference the trigger passes by default. The domain subset is chosen at upload
# time (which cases you export), not per-trigger.
DEFAULT_DATASET_REFERENCE = f"{SPEC_NAME}-dataset@production"


def _short_sha() -> str:
    """Return ``git rev-parse --short HEAD`` for the cookbook repo.

    Used as the default push version so each ``main`` push lands a
    fresh, immutable SpecVersion without manual bumps. Falls back to
    ``"dev"`` if git is unavailable.
    """
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


# Requirements the sandbox image already provides without an index install:
# ``rilixai`` (baked in by the build worker) and ``cookbook-common`` (a
# workspace member installed by the bundle-root ``pip install /spec``). Both
# are matched by normalized distribution name (``_`` / ``-`` insensitive).
_BUNDLE_PROVIDED_PREFIXES = ("rilixai",)
_BUNDLE_PROVIDED_NAMES = ("cookbook-common",)


def _is_bundle_provided(dep: str) -> bool:
    """True if ``dep`` is already in the image and must not be ``--pip-install``ed."""
    name = re.split(r"[<>=!~;\[ ]", dep.strip(), maxsplit=1)[0].strip().lower().replace("_", "-")
    return name.startswith(_BUNDLE_PROVIDED_PREFIXES) or name in _BUNDLE_PROVIDED_NAMES


def _member_pip_deps() -> list[str]:
    """Read the apex_agents workspace member's runtime deps from its pyproject.

    The rilixai build worker only sees the *bundle root* pyproject
    when it runs ``pip install /spec``; the workspace member's
    pyproject is invisible to it. So we shovel the member's deps in
    via ``--pip-install`` explicitly. Reading them here keeps the dep
    list from drifting between ``apex_agents/pyproject.toml`` and the
    push invocation.

    ``rilixai`` is stripped because rilixai's build worker bakes its
    own pinned wheel into every spec image — customer pins for
    rilixai are rejected. ``cookbook-common`` is stripped because it is
    a workspace member with no index release: the bundle-root ``pip
    install /spec`` already installs it (the root pyproject's setuptools
    package list includes ``cookbook_common*``), so passing it to
    ``--pip-install`` would send the build worker looking for a
    nonexistent PyPI release.
    """
    data = tomllib.loads(MEMBER_PYPROJECT.read_text())
    deps = data["project"]["dependencies"]
    return [d for d in deps if not _is_bundle_provided(d)]


def push_image(version: str) -> None:
    """Run ``rilixai push`` to build a new Modal image for this spec version."""
    deps = _member_pip_deps()
    pip_install_args: list[str] = []
    for dep in deps:
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
        *pip_install_args,
        str(SPEC_TARGET),
    ]
    print(f"\n→ {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)


def promote_image(version: str) -> None:
    """Run ``rilixai spec promote`` to point ``@production`` at the given version."""
    cmd = [
        "uv",
        "run",
        "rilixai",
        "spec",
        "promote",
        SPEC_NAME,
        version,
    ]
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
        # The optimizer reads its cases from this uploaded JSONL dataset — the
        # migrated spec no longer loads data itself, so a run with no dataset
        # reference fails at startup. ``name@production`` resolves server-side
        # to the currently promoted dataset revision.
        dataset_ref=dataset_reference,
        config={
            # GEPA per-run knobs (consumed by rilixai's sandbox runtime).
            "max_metric_calls": max_metric_calls,
            "reflection_minibatch_size": 3,
            "reflection_model": "openai/gpt-4.1",
            "seed": 0,
            # APEX-Agents cookbook knobs (consumed by build_spec in spec.py).
            # The domain subset + train/val split come from the uploaded
            # dataset, so no domain/train_size/val_size/val_worlds knobs here.
            "task_model": "openai/gpt-4.1-mini-2025-04-14",
            "task_temperature": 0.0,
            "judge_model": "gemini/gemini-2.5-flash",
            "max_steps": 60,
            "cost_limit": 3.0,
        },
    )
    return str(response["id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--build",
        action="store_true",
        help="Push the current apex_agents source as a new spec image, then (by default) promote it to @production.",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="With --build, push without promoting. Lets you smoke-test a new build before flipping production.",
    )
    parser.add_argument(
        "--no-trigger",
        action="store_true",
        help=(
            "Skip the optimization-run trigger after build/promote. Used by the CI "
            "``push-spec.yml`` workflow, which only needs to ship the spec image and "
            "flip @production — not spend tokens on a smoke run."
        ),
    )
    parser.add_argument(
        "--version",
        default=None,
        help=(
            "Spec version to push (used only with --build). Defaults to v<short_sha> from git so each "
            "main push lands an immutable SpecVersion automatically."
        ),
    )
    parser.add_argument(
        "--spec",
        default=DEFAULT_SPEC_REFERENCE,
        help=(
            f"rilixai spec reference for the trigger. Defaults to {DEFAULT_SPEC_REFERENCE}, which rilixai "
            "resolves server-side to the currently promoted version."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_REFERENCE,
        help=(
            f"rilixai dataset reference the run sources cases from. Defaults to "
            f"{DEFAULT_DATASET_REFERENCE} (upload it first with "
            f"`rilixai dataset upload --name {SPEC_NAME}-dataset <jsonl-dir>`). "
            "Override with apex-agents-dataset@<revision> to pin a revision."
        ),
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=50,
        help="GEPA metric-call budget — primary cost knob. Default: 50 (smoke).",
    )
    args = parser.parse_args()

    # Validate the trigger-path env vars *before* any build/promote work so a
    # misconfigured ``--build`` (without ``--no-trigger``) can't half-complete
    # — push a real image, flip @production, then die at the trigger step.
    # ``--no-trigger`` skips the RilixAIClient call entirely, so its env vars
    # aren't needed; ``rilixai push`` / ``spec promote`` use the rilixai CLI's
    # own auth.
    api_key: str | None = None
    base_url: str | None = None
    if not args.no_trigger:
        api_key = os.environ.get("RILIXAI_API_KEY")
        base_url = os.environ.get("RILIXAI_API_BASE_URL")
        if not api_key or not base_url:
            print("error: RILIXAI_API_KEY and RILIXAI_API_BASE_URL must be set.", file=sys.stderr)
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
    client = RilixAIClient(base_url=base_url, api_key=api_key)
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
