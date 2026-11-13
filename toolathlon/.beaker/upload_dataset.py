"""Upload Toolathlon dataset splits (full 108-task benchmark, its no-external-account subset, or 3-task quickstart)
without persisting local JSONL."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from toolathlon_setup import build_task_row, prepare_toolathlon

from toolathlon_baseline import QUICKSTART_TASKS
from toolathlon_baseline.catalog import task_catalog


QUICKSTART_TRAIN_TASKS = ("find-alita-paper", "excel-market-research")
QUICKSTART_TEST_TASKS = ("interview-report",)
# MCP servers backed by third-party accounts (tokens in upstream configs/token_key_session.py).
# Canvas, emails, WooCommerce and k8s are local containers and need no account.
EXTERNAL_ACCOUNT_SERVERS = frozenset(
    {
        "github",
        "google-cloud",
        "google_calendar",
        "google_forms",
        "google_map",
        "google_sheet",
        "huggingface",
        "notion",
        "snowflake",
        "wandb",
        "yahoo-finance",
        "youtube",
        "youtube-transcript",
    }
)


def _full_split_tasks(source: Path, *, no_external_accounts: bool = False) -> tuple[list[str], list[str]]:
    """Return a deterministic 2/3 train and 1/3 test split of the official tasks (72/36 for all 108)."""
    remaining = [
        entry["id"]
        for entry in task_catalog(source)
        if entry["id"] not in QUICKSTART_TASKS
        and not (no_external_accounts and EXTERNAL_ACCOUNT_SERVERS & set(entry["mcp_servers"]))
    ]
    ordered = [*QUICKSTART_TASKS, *remaining]
    train_tasks = [task for idx, task in enumerate(ordered) if idx % 3 != 2]
    test_tasks = [task for idx, task in enumerate(ordered) if idx % 3 == 2]
    return train_tasks, test_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Upload all 108 Toolathlon tasks with a 2/3 train (72) and 1/3 test (36) split (default).",
    )
    parser.add_argument(
        "--quickstart",
        action="store_true",
        help="Upload the 3 native quickstart tasks instead (2 train, 1 test).",
    )
    parser.add_argument(
        "--no-external-accounts",
        action="store_true",
        help="Upload only the 49 tasks whose MCP servers need no third-party account (33 train, 16 test).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Dataset name override (defaults to toolathlon-full or toolathlon-quickstart).",
    )
    args = parser.parse_args()
    if args.full and args.quickstart:
        parser.error("--full and --quickstart are mutually exclusive")
    if args.quickstart and args.no_external_accounts:
        parser.error("--quickstart and --no-external-accounts are mutually exclusive")

    _, source = prepare_toolathlon()
    if args.quickstart:
        train_tasks = list(QUICKSTART_TRAIN_TASKS)
        test_tasks = list(QUICKSTART_TEST_TASKS)
        dataset_name = args.name or "toolathlon-quickstart"
    elif args.no_external_accounts:
        train_tasks, test_tasks = _full_split_tasks(source, no_external_accounts=True)
        dataset_name = args.name or "toolathlon-no-external-accounts"
    else:
        train_tasks, test_tasks = _full_split_tasks(source)
        dataset_name = args.name or "toolathlon-full"

    splits = {
        "train": [build_task_row(source, task) for task in train_tasks],
        "test": [build_task_row(source, task) for task in test_tasks],
    }
    if {row["id"] for row in splits["train"]} & {row["id"] for row in splits["test"]}:
        raise ValueError("Training and evaluation tasks must be disjoint")

    with tempfile.TemporaryDirectory(prefix="beaker-toolathlon-dataset-") as temporary:
        directory = Path(temporary)
        for split, selected in splits.items():
            (directory / f"{split}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in selected),
                encoding="utf-8",
            )
        subprocess.run(
            [
                "beaker",
                "run",
                "smoke",
                "--strict",
                "--integration-id",
                "toolathlon_baseline",
                "--config",
                json.dumps({"local_dataset_path": temporary}),
            ],
            check=True,
        )
        subprocess.run(
            [
                "beaker",
                "dataset",
                "upload",
                temporary,
                "--agent",
                "toolathlon",
                "--name",
                dataset_name,
                "--total-count",
                str(sum(len(rows) for rows in splits.values())),
                "--split",
                f"train={len(splits['train'])}",
                "--split",
                f"test={len(splits['test'])}",
                "--json",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
