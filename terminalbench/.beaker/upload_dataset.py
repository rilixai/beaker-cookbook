"""Upload the development subset or full evaluation set to an explicit Beaker API."""

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
    parser.add_argument("--name")
    parser.add_argument("--full", action="store_true", help="Upload all 89 benchmark tasks as test; no training split")
    args = parser.parse_args()
    name = args.name or ("terminalbench-2.0-full" if args.full else "terminalbench-small")
    with tempfile.TemporaryDirectory(prefix="terminalbench-dataset-") as temporary:
        counts = write_dataset(Path(temporary), full=args.full)
        split_args = [arg for split, count in counts.items() for arg in ("--split", f"{split}={count}")]
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
                name,
                *split_args,
                "--json",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
