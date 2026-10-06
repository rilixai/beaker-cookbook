"""Keep AppWorld's process-global state private to each Beaker case."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import tempfile
from pathlib import Path

from beaker import CaseResult, RetryableCaseError
from beaker.tracing import capture_identity
from beaker.tracing.projection import parse_jsonl


WORKER = Path(__file__).with_name("appworld_case_worker.py")


def _adopt_trace(trace, directory: Path) -> None:
    """Use the SDK's external-span adoption path, including spilled content."""
    session = getattr(trace, "session", None)
    if session is None:
        return
    receipts = list((directory / "traces" / "captures").glob("*.receipt.json"))
    if not receipts:
        session.omissions.append("AppWorld subprocess exited without a trace receipt")
        return
    receipt = json.loads(receipts[0].read_text())
    # Re-store artifacts before deleting the worker's temporary directory. Their
    # content hashes stay unchanged, preserving references from adopted spans.
    for ref in json.loads((directory / "artifacts.json").read_text()):
        content = ref.get("content") if ref.get("inline") else Path(ref["uri"]).read_bytes()
        if ref.get("kind") == "evidence":
            if isinstance(content, bytes):
                content = content.decode() if ref["media_type"].startswith("text/") else json.loads(content)
            trace.evidence(ref["name"], content)
        else:
            trace.artifact(
                ref["name"],
                content,
                media_type=ref["media_type"],
                filename=ref.get("filename"),
                required=ref.get("required", False),
            )
    for batch in parse_jsonl(
        receipts[0].with_name(receipts[0].name.replace(".receipt.json", ".otlp.jsonl")).read_bytes()
    ):
        session.manager.adopt(session, batch)
    session.warnings.extend(receipt.get("warnings", []))
    session.omissions.extend(receipt.get("omissions", []))


async def _stop(process: asyncio.subprocess.Process) -> None:
    # A separate process group also lets cancellation clean up agent descendants.
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.returncode is None:
        process.kill()
    await process.wait()


async def run_isolated_case(*, case_input, runtime) -> CaseResult:
    with tempfile.TemporaryDirectory(prefix="appworld-case-") as temporary:
        directory = Path(temporary)
        (directory / "request.json").write_text(
            json.dumps(
                {
                    "input": case_input,
                    "model": runtime.model,
                    "identity": capture_identity(runtime.trace),
                }
            )
        )
        # File-backed output avoids pipe deadlocks and unbounded in-memory logs.
        with (directory / "worker.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(WORKER),
                str(directory),
                cwd=WORKER.parent.parent,
                stdout=log,
                stderr=log,
                start_new_session=os.name == "posix",
            )
            try:
                await process.wait()
            finally:
                await _stop(process)
                try:
                    _adopt_trace(runtime.trace, directory)
                except Exception as exc:
                    session = getattr(runtime.trace, "session", None)
                    if session is not None:
                        session.omissions.append(f"AppWorld subprocess trace import failed: {exc}")
        response_path = directory / "response.json"
        if process.returncode or not response_path.exists():
            with (directory / "worker.log").open("rb") as log:
                log.seek(max(0, log.seek(0, 2) - 8000))
                detail = log.read().decode(errors="replace")
            raise RuntimeError(f"AppWorld subprocess exited with code {process.returncode}: {detail}")
        response = json.loads(response_path.read_text())
        if "error" in response:
            runtime.trace.artifact("appworld-error", response["traceback"], media_type="text/plain")
            error_type = RetryableCaseError if response["retryable"] else RuntimeError
            raise error_type(response["error"])
        return CaseResult(output=response["output"], output_kind=response["output_kind"])
