"""Convert frozen AutomationBench splits to a temporary Beaker JSONL dataset and upload it.

Writes nothing into the repository. Invoke from the automationbench project root.
The dataset is uploaded to the agent recorded in ``.beaker/beaker.yaml`` (the key
``beaker agent setup`` wrote), unless ``--agent`` overrides it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import yaml

from automationbench_skills.data.tasks import PUBLIC_DOMAINS, Sample, load_split


TRAIN_PER_DOMAIN = 6
TEST_PER_DOMAIN = 3
DATASET_NAME = "automationbench-skills-quickstart"
BEAKER_YAML = Path(__file__).resolve().parent / "beaker.yaml"


def configured_agent_key() -> str:
    config = yaml.safe_load(BEAKER_YAML.read_text(encoding="utf-8"))
    integrations = config["integrations"]
    integration = integrations[config.get("default_integration") or next(iter(integrations))]
    agent_key = integration.get("agent_key")
    if not agent_key:
        raise RuntimeError(f"no agent_key in {BEAKER_YAML}; run `beaker agent setup` first or pass --agent")
    return str(agent_key)


def _user_prompt(sample: Sample) -> str:
    return "\n\n".join(
        str(message.get("content") or "") for message in sample.prompt if message.get("role") == "user"
    ).strip()


def _take_per_domain(split: str, per_domain: int) -> list[dict[str, object]]:
    by_domain: dict[str, list[Sample]] = defaultdict(list)
    for sample in load_split(split):
        by_domain[sample.domain].append(sample)
    rows: list[dict[str, object]] = []
    for domain in PUBLIC_DOMAINS:
        for sample in by_domain[domain][:per_domain]:
            rows.append(
                {
                    "id": sample.task_name,
                    # The Integration loads the task by name; ``prompt`` is what the
                    # agent was asked, so the case view shows the ask next to
                    # the assertion checks.
                    "input": {"task_name": sample.task_name, "prompt": _user_prompt(sample)},
                    # The task's assertions are what "correct" means for this
                    # case; ``score_case`` evaluates them against the end state
                    # and emits one Check per assertion.
                    "expected": {"assertions": sample.info["assertions"]},
                    "metadata": {"domain": domain, "source_split": split},
                    "group_key": domain,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default=None, help="agent key; defaults to agent_key in .beaker/beaker.yaml")
    args = parser.parse_args()
    agent_key = args.agent or configured_agent_key()
    train_rows = _take_per_domain("train", TRAIN_PER_DOMAIN)
    test_rows = _take_per_domain("test", TEST_PER_DOMAIN)
    with tempfile.TemporaryDirectory(prefix="beaker-dataset-") as temp_dir:
        dataset_dir = Path(temp_dir)
        splits = {"train": train_rows, "test": test_rows}
        for split_name, rows in splits.items():
            split_path = dataset_dir / f"{split_name}.jsonl"
            with split_path.open("w", encoding="utf-8") as output:
                for row in rows:
                    output.write(json.dumps(row) + "\n")
        beaker = shutil.which("beaker")
        if not beaker:
            raise RuntimeError("beaker CLI is not on PATH")
        upload = subprocess.run(
            [
                beaker,
                "dataset",
                "upload",
                str(dataset_dir),
                "--name",
                DATASET_NAME,
                "--agent",
                agent_key,
                "--total-count",
                str(sum(len(rows) for rows in splits.values())),
                "--split",
                f"train={len(train_rows)}",
                "--split",
                f"test={len(test_rows)}",
                "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    print(upload.stdout)


if __name__ == "__main__":
    main()
