"""Convert HealthBench's public JSONL into Beaker train/test splits and upload them.

Usage (from healthbench/): uv run python .beaker/build_dataset.py --name healthbench --subset full
Small dataset: add --limit 120 to use the same random sample as both train and test.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import tempfile
from pathlib import Path

from healthbench.data.dataset import load_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--subset", default="full", choices=["full", "hard", "consensus"])
    parser.add_argument("--agent", default="healthbench")
    parser.add_argument("--limit", type=int, help="Use this many cases as both train and test.")
    args = parser.parse_args()

    if args.limit:
        sample = load_records(args.subset, "all", limit=args.limit)
        # Case ids must be unique across the dataset, so the test copy gets its own ids.
        test_copy = [dataclasses.replace(r, prompt_id=f"{r.prompt_id}-test") for r in sample]
        splits = {"train": sample, "test": test_copy}
    else:
        splits = {split: load_records(args.subset, split) for split in ("train", "test")}
    with tempfile.TemporaryDirectory(prefix="beaker-healthbench-") as temp_dir:
        for split, records in splits.items():
            with (Path(temp_dir) / f"{split}.jsonl").open("w", encoding="utf-8") as output:
                for record in records:
                    output.write(json.dumps(dataclasses.asdict(record)) + "\n")
        subprocess.run(
            [
                "beaker",
                "dataset",
                "upload",
                temp_dir,
                "--name",
                args.name,
                "--total-count",
                str(sum(len(r) for r in splits.values())),
                *(arg for split, r in splits.items() for arg in ("--split", f"{split}={len(r)}")),
                "--agent",
                args.agent,
                "--json",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
