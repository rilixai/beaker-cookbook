"""Opt-in real Toolathlon/MCP checks with a deterministic local model endpoint.

Set TOOLATHLON_TEST_CACHE to an already prepared official cache. These tests
never call a paid model or download benchmark assets during normal pytest.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from beaker.tracing import local_capture


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".beaker"))
import beaker_integration as integration  # noqa: E402


@pytest.mark.parametrize(
    "task,limit",
    [
        ("excel-market-research", None),
        ("interview-report", None),
        ("find-alita-paper", None),
        ("excel-market-research", 1),
    ],
)
def test_real_task_runner_tools_context_and_native_traces(
    task: str, limit: int | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = os.environ.get("TOOLATHLON_TEST_CACHE")
    if not configured:
        pytest.skip("Set TOOLATHLON_TEST_CACHE to run real upstream/MCP tests without paid models")
    cache = Path(configured)
    monkeypatch.setenv("BEAKER_TOOLATHLON_CACHE", str(cache))
    monkeypatch.setenv("TOOLATHLON_MODEL", "local-native-test")
    monkeypatch.setenv("TOOLATHLON_OPENAI_API_KEY", "local-test")
    monkeypatch.setattr(integration, "DEFAULT_TIMEOUT_SECONDS", 60)
    if limit is not None:
        monkeypatch.setattr(integration, "DEFAULT_MAX_STEPS", limit)
    requests: list[dict] = []
    errors: list[str] = []

    def reply(request: dict) -> dict:
        requests.append(request)
        turn = len(requests)
        tools = request["tools"]
        previous = [msg for msg in request["messages"] if msg["role"] == "tool"]
        workspace = re.search(r"Accessible workspace directory: ([^\n]+)", request["messages"][0]["content"])[1]

        def call(fragment: str, arguments: dict, call_id: str) -> dict:
            name = next(tool["function"]["name"] for tool in tools if fragment in tool["function"]["name"])
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
                ],
            }

        if task == "excel-market-research" and turn == 1:
            reference = cache / "upstream/tasks/finalpool/excel-market-research/groundtruth_workspace"
            reference /= "segment_growth_rates.xlsx"
            code = (
                "from pathlib import Path\ntry:\n"
                f" Path({str(reference)!r}).read_bytes()\n"
                "except PermissionError:\n Path('probe.txt').write_text('reference denied')\n"
                " print('reference denied')\nelse:\n raise AssertionError('reference leaked')\n"
            )
            return call("python_execute", {"code": code, "filename": "probe.py", "timeout": 10}, "python-probe")
        if task == "excel-market-research" and turn == 2:
            assert "reference denied" in str(previous)
            return call(
                "manage_context",
                {
                    "action": "truncate",
                    "method": "keep_recent_turns",
                    "value": 1,
                    "preserve_system": True,
                    "preserve_first_user_input": True,
                },
                "manage-probe",
            )
        if task == "excel-market-research" and turn == 3:
            assert "python-probe" not in str(request["messages"])
            assert "manage-probe" in str(request["messages"])
        if task == "interview-report" and turn == 1:
            return call("word_create_document", {"filename": f"{workspace}/probe.docx"}, "word-probe")
        if task == "find-alita-paper" and turn == 1:
            return call(
                "filesystem_write_file", {"path": f"{workspace}/probe.txt", "content": "MCP probe"}, "fs-probe"
            )
        return {"role": "assistant", "content": "Task completed."}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                message = reply(request)
            except Exception as exc:
                errors.append(repr(exc))
                message = {"role": "assistant", "content": "Stop after test failure."}
            index = min(len(requests) - 1, 2)
            tokens, cached = [10, 90, 7][index], [4, 2, 0][index]
            response = {
                "id": "local-native-test",
                "object": "chat.completion",
                "created": 1,
                "model": request["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": tokens,
                    "completion_tokens": 1,
                    "total_tokens": tokens + 1,
                    "prompt_tokens_details": {"cached_tokens": cached},
                },
            }
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("TOOLATHLON_OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    try:
        with local_capture(tmp_path / "trace", case_id=task, candidate_id="native-sdk") as capture:
            result = asyncio.run(
                integration.run_case(
                    case_input={"task": task, "instruction": "official task", "mcp_servers": ["filesystem"]},
                    runtime=SimpleNamespace(model=None, trace=capture.trace),
                )
            )
        assert not errors
        if limit is not None:
            assert result.output["agent_status"] == "max_turns_reached"
            assert len(requests) == 1
            assert capture.receipt.counts["model_calls"] == 1
            assert not capture.receipt.omissions
            return
        assert result.output["agent_status"] == "success"
        assert result.output["agent_framework"] == "openai-agents==0.20.0"
        capture.require(model_io=True, tool_io=True, zero_drops=True)
        assert not capture.receipt.omissions
        assert capture.receipt.counts["model_calls"] == len(requests)
        assert capture.receipt.projection["token_usage"]["cached_input_tokens"] == 6
        files = result.context["workspace_files"]
        if task == "excel-market-research":
            assert len(requests) == 3
            assert capture.receipt.projection["token_usage"]["input_tokens"] == 107
            assert base64.b64decode(files["probe.txt"]) == b"reference denied"
        elif task == "interview-report":
            assert base64.b64decode(files["probe.docx"]).startswith(b"PK")
        else:
            assert base64.b64decode(files["probe.txt"]) == b"MCP probe"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
