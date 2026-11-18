"""Package pinned source files and bounded text for read-only dataset reviews."""

from __future__ import annotations

import gzip
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from harvey_lab.agent.workspace import extract_text
from harvey_lab.config import HARVEY_LABS_COMMIT


MAX_TEXT_BYTES = 32 * 1024 * 1024
TEXT_FORMATS = {".txt", ".md", ".csv", ".json", ".xml", ".html"}
DOCUMENT_FORMATS = {".pdf", ".docx", ".xlsx", ".xlsm", ".pptx", ".eml"}


def package_documents(task_dir: Path, task_id: str, output: Path) -> dict[str, Any]:
    """Keep originals separate from text; never run macros, formulas or customer code."""
    if (task_dir / "documents").is_symlink():
        raise ValueError("Source documents must not contain symbolic links")
    key = hashlib.sha256(task_id.encode()).hexdigest()
    folder = output / "review-evidence"
    folder.mkdir(exist_ok=True)
    archive = folder / f"{key}.zip"
    documents: dict[str, Any] = {}
    remaining = MAX_TEXT_BYTES - 1024 * 1024
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted((task_dir / "documents").rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Source documents must not contain symbolic links: {path}")
            if not path.is_file():
                continue
            name = path.relative_to(task_dir / "documents").as_posix()
            raw = path.read_bytes()
            # Stable ZIP metadata makes repeated uploads reproducible.
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, raw)
            suffix = path.suffix.lower()
            supported = suffix in TEXT_FORMATS | DOCUMENT_FORMATS
            text = extract_text(path) if supported and remaining > 0 else ""
            limitations = []
            status = "extracted"
            if not supported:
                status = "unsupported"
                limitations.append("No text reader for this file type.")
            elif not text.strip() or text.startswith(("Could not extract", "Could not parse")):
                status = "unreadable"
                limitations.append("Text extraction failed or returned no readable text.")
            if suffix in DOCUMENT_FORMATS:
                limitations.append("Text representation only; images and rendered layout are not assessed.")
            if suffix in {".xlsx", ".xlsm"}:
                limitations.append(
                    "At most 200 rows and 40 columns per sheet; formulas and conditional formatting are not evaluated."
                )
                status = "partial" if status == "extracted" else status
            if suffix == ".eml":
                limitations.append("Email body and headers only; attachments are retained in the original file.")
            if "... (truncated" in text:
                status = "partial"
                limitations.append("The text reader truncated this document.")
            encoded = text.encode()
            if len(encoded) > remaining or remaining <= 0:
                text = encoded[: max(0, remaining)].decode("utf-8", errors="ignore")
                status = "partial"
                limitations.append("The task exceeded the review text size limit.")
            remaining -= len(text.encode())
            documents[name] = {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "status": status,
                "limitations": limitations,
                "text": text,
            }
    payload = json.dumps(
        {
            "source_commit": HARVEY_LABS_COMMIT,
            "task_id": task_id,
            "reader": "harvey_lab.agent.workspace.extract_text",
            "documents": documents,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if len(payload) > MAX_TEXT_BYTES:
        raise ValueError(f"Document index exceeds the per-task evidence limit: {task_id}")
    encoded = gzip.compress(payload, mtime=0)
    path = folder / f"{key}.json.gz"
    path.write_bytes(encoded)
    with archive.open("rb") as stream:
        archive_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "artifact_path": path.relative_to(output).as_posix(),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "decoded_bytes": len(payload),
        "source_commit": HARVEY_LABS_COMMIT,
        "archive_path": archive.relative_to(output).as_posix(),
        "archive_sha256": archive_digest,
    }
