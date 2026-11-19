"""Uploading task rows must not package or parse their source documents."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def test_upload_keeps_pinned_manifest_without_document_bundles(tmp_path, monkeypatch):
    script = Path(__file__).parents[1] / ".beaker" / "upload_splits.py"
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location("upload_resources_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "read_split", lambda split: [f"contracts/{split}"])
    monkeypatch.setattr(module, "sample_train", lambda train, *args: train)
    monkeypatch.setattr(module, "distribution_report", lambda *args: {})
    monkeypatch.setattr(module, "ensure_task_dirs", lambda ids: tmp_path)
    monkeypatch.setattr(module.shutil, "which", lambda name: "/fake/beaker")
    (tmp_path / "unreadable.docx").write_bytes(b"not a document")
    monkeypatch.setattr(
        module,
        "load_records",
        lambda root, task_ids: [SimpleNamespace(task_id=task_id, practice_area="contracts") for task_id in task_ids],
    )
    monkeypatch.setattr(
        module,
        "row_payload",
        lambda record: (
            {"task_id": record.task_id, "title": "Example", "instructions": "Review", "deliverables": {}},
            {"criteria": [{"id": "one", "title": "One", "match_criteria": "Correct", "deliverables": []}]},
        ),
    )
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[1:3] == ["run", "smoke"]
        config = json.loads(command[command.index("--config") + 1])
        dataset = Path(config["local_dataset_path"])
        assert {path.name for path in dataset.iterdir()} == {"train.jsonl", "test.jsonl", "manifest.json"}
        manifest = json.loads((dataset / "manifest.json").read_text())
        assert manifest["source"] == "harveyai/harvey-labs"
        assert manifest["commit"] == module.HARVEY_LABS_COMMIT
        assert "review_documents" not in manifest
        for split in ("train", "test"):
            row = json.loads((dataset / f"{split}.jsonl").read_text())
            assert row["id"] == f"contracts/{split}"
            assert manifest["metadata"]["review_resources"][row["id"]] == {
                "repository": "harveyai/harvey-labs",
                "commit": module.HARVEY_LABS_COMMIT,
                "directory": f"tasks/contracts/{split}/documents",
            }
            assert row["input"]["commit"] == manifest["commit"]

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", [str(script), "--smoke-only", "--train-limit", "1", "--test-limit", "1"])
    module.main()
    assert len(calls) == 1
