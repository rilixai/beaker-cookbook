"""Upload the real fixed split. Require an explicit API URL to avoid wrong-environment writes."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from benchmark_data import write_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--name", default="terminalbench-small")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="terminalbench-dataset-") as temporary:
        write_dataset(Path(temporary))
        subprocess.run(
            [
                str(Path(sys.executable).with_name("beaker")),
                "--base-url",
                args.base_url,
                "dataset",
                "upload",
                temporary,
                "--agent",
                args.agent,
                "--name",
                args.name,
                "--split",
                "train=8",
                "--split",
                "test=4",
                "--json",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
