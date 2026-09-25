"""Upload the pinned, labeled OfficeQA Pro splits without persisting JSONL."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from officeqa import load_split


def _row(record: object) -> dict[str, str]:
    return {
        "uid": str(getattr(record, "uid")),
        "question": str(getattr(record, "question")),
        "answer": str(getattr(record, "answer")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--name", default="officeqa-pro")
    args = parser.parse_args()

    splits = {name: [_row(record) for record in load_split(name)] for name in ("train", "test")}
    total = sum(len(rows) for rows in splits.values())

    with tempfile.TemporaryDirectory(prefix="beaker-officeqa-") as temp_dir:
        dataset_dir = Path(temp_dir)
        for split_name, rows in splits.items():
            with (dataset_dir / f"{split_name}.jsonl").open("w", encoding="utf-8") as output:
                for row in rows:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")

        command = [
            "beaker",
            "dataset",
            "upload",
            str(dataset_dir),
            "--name",
            args.name,
            "--total-count",
            str(total),
            "--agent",
            args.agent,
            "--json",
        ]
        for split_name, rows in splits.items():
            command.extend(("--split", f"{split_name}={len(rows)}"))
        completed = subprocess.run(command, check=True, text=True, capture_output=True)

    artifact = json.loads(completed.stdout)
    print(
        json.dumps(
            {
                "dataset": f"{artifact['artifact_key']}@{artifact['dataset_revision']}",
                "artifact_id": artifact.get("artifact_id"),
                "total_count": total,
                "splits": {name: len(rows) for name, rows in splits.items()},
            }
        )
    )


if __name__ == "__main__":
    main()
