"""Build (optional) + promote + trigger one hotpotqa-agent run on rilixai's Modal sandbox.

The sibling of ``cli.py`` for the sandbox path. ``cli.py`` runs the local
laptop loop; this file pushes the source as a Modal image (via
``rilixai push``), optionally promotes the freshly-pushed row to
``hotpotqa-agent@production``, then queues a run via
:class:`rilixai.RilixAIClient` against the production reference.

Version strategy (mirrors rilix/rilix PR #1381):

* No version pinned in ``@spec(...)``. ``--build`` defaults the push
  version to ``v<short_sha>`` so each ``main`` push lands an immutable
  SpecVersion automatically.
* After push, ``--build`` also calls ``rilixai spec promote`` to point
  ``hotpotqa-agent@production`` at the just-pushed version. Pass
  ``--no-promote`` to push without promoting (smoke / regression).
* The trigger defaults to ``hotpotqa-agent@production``, which rilixai
  resolves server-side to the currently promoted version (falls back
  to the latest READY version when nothing is promoted). Override
  ``--spec hotpotqa-agent@v<sha>`` to pin a specific build for
  regression debugging.

Typical workflows:

    # Build + promote + trigger in one shot (canonical local dev flow):
    uv run hotpotqa/sandbox.py --build

    # Build + promote only, no trigger (the CI ``push-spec.yml`` flow —
    # ships the image and flips @production without spending LLM tokens):
    uv run hotpotqa/sandbox.py --build --no-trigger

    # Trigger only (uses whatever's currently @production):
    uv run hotpotqa/sandbox.py

    # Pin a specific version (smoke / regression):
    uv run hotpotqa/sandbox.py --spec hotpotqa-agent@v1a2b3c4

    # Push without promoting (lets you smoke-test a new build before flipping prod):
    uv run hotpotqa/sandbox.py --build --no-promote --spec hotpotqa-agent@v1a2b3c4

Required env vars (load via .env or export):
    RILIXAI_API_BASE_URL   — API Gateway URL from the RilixaiApiStack CDK output
    RILIXAI_API_KEY        — control-plane credential
    OPENAI_API_KEY         — needed locally only for ``--build`` (the rilixai
                             push CLI does not require it; bound at the project
                             level inside the sandbox)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from dotenv import load_dotenv
from rilixai import RilixAIClient


load_dotenv()


# Cookbook layout the script is hardwired against. This file lives at
# ``hotpotqa/sandbox.py``; parents[1] is the cookbook repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
MEMBER_PYPROJECT = REPO_ROOT / "hotpotqa" / "pyproject.toml"
SPEC_TARGET = REPO_ROOT / "hotpotqa" / "optimization" / "spec.py"
SPEC_NAME = "hotpotqa-agent"
SCOPE_KEY = "hotpotqa-agent"
TASK_TYPE = "hotpotqa_pydantic_agent"
DEFAULT_SPEC_REFERENCE = f"{SPEC_NAME}@production"


def _short_sha() -> str:
    """Return ``git rev-parse --short HEAD`` for the cookbook repo.

    Used as the default push version so each ``main`` push lands a
    fresh, immutable SpecVersion without manual bumps. Falls back to
    ``"dev"`` if git is unavailable (rare — but keeps offline dev
    flows working).
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


def _member_pip_deps() -> list[str]:
    """Read the hotpotqa workspace member's runtime deps from its pyproject.

    The rilixai build worker only sees the *bundle root* pyproject when it
    runs ``pip install /spec``; the workspace member's pyproject is
    invisible to it. So we shovel the member's deps in via ``--pip-install``
    explicitly. Reading them here keeps the dep list from drifting between
    ``hotpotqa/pyproject.toml`` and the push invocation.

    ``rilixai`` is stripped because rilixai's build worker bakes its own
    pinned wheel into every spec image (see ``RESERVED_PIP_INSTALL_NAMES``
    in ``rilixai/_image_install_validation.py``) — customer pins for
    rilixai are rejected.
    """
    data = tomllib.loads(MEMBER_PYPROJECT.read_text())
    deps = data["project"]["dependencies"]
    return [d for d in deps if not d.lstrip().lower().startswith("rilixai")]


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
    """Run ``rilixai spec promote`` to point ``@production`` at the given version.

    Equivalent to rilix's ``make promote-spec`` step. Safe to invoke
    standalone with an older ``--version`` to roll back to a known-good
    build. ``rilixai spec demote NAME`` clears the promotion and falls
    back to "latest READY".
    """
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
    train_size: int,
    val_size: int,
    max_metric_calls: int,
    retrieval_mode: str,
) -> str:
    response = client.create_optimization_run(
        task_type=TASK_TYPE,
        spec=spec_reference,
        scope_key=SCOPE_KEY,
        config={
            # GEPA per-run knobs (consumed by rilixai's sandbox runtime).
            "max_metric_calls": max_metric_calls,
            "reflection_minibatch_size": 3,
            "reflection_model": "openai/gpt-4.1",
            "seed": 0,
            # HotpotQA cookbook knobs (consumed by build_spec in spec.py).
            "retrieval_mode": retrieval_mode,
            "retrieve_k": 7,
            "max_iters": 8,
            "train_size": train_size,
            "val_size": val_size,
            "pydantic_agent_model": "openai:gpt-4.1-mini",
            "task_temperature": 0.0,
        },
    )
    return str(response["id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--build",
        action="store_true",
        help="Push the current hotpotqa source as a new spec image, then (by default) promote it to @production.",
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
            "Spec version to push (used only with --build). Defaults to v<short_sha> "
            "from git so each main push lands an immutable SpecVersion automatically."
        ),
    )
    parser.add_argument(
        "--spec",
        default=DEFAULT_SPEC_REFERENCE,
        help=(
            f"rilixai spec reference for the trigger. Defaults to {DEFAULT_SPEC_REFERENCE}, "
            "which rilixai resolves server-side to the currently promoted version. "
            "Override with hotpotqa-agent@v<sha> to pin a specific build."
        ),
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=25,
        help="HotpotQA train cases. Paper uses 150. Default: 25.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=25,
        help="HotpotQA validation cases. Paper uses 300. Default: 25.",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=100,
        help="GEPA metric-call budget — primary cost knob. Default: 100 (smoke); paper uses 6871.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=("distractor", "fullwiki"),
        default="distractor",
        help="``distractor`` (default) avoids the 5GB fullwiki download per cold start.",
    )
    args = parser.parse_args()

    # Fail fast on missing trigger-path env vars *before* doing any
    # build/promote work, so a misconfigured ``--build`` (without
    # ``--no-trigger``) doesn't half-complete — push a real image,
    # flip @production, then die at the trigger step. ``--no-trigger``
    # skips the RilixAIClient call entirely, so its env vars aren't
    # needed; ``rilixai push`` + ``rilixai spec promote`` use their
    # own auth via the rilixai CLI.
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
        train_size=args.train_size,
        val_size=args.val_size,
        max_metric_calls=args.max_metric_calls,
        retrieval_mode=args.retrieval_mode,
    )
    print(f"queued run: {run_id} (spec={args.spec})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
