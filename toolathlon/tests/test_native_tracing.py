from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from beaker.tracing import local_capture
from beaker.tracing.projection import parse_jsonl


INTEGRATION = Path(__file__).resolve().parents[1] / ".beaker"
sys.path.insert(0, str(INTEGRATION))
from toolathlon_native_trace import NativeTrace  # noqa: E402


NATIVE_CHILD = """
import asyncio, os
from pathlib import Path
from agents import generation_span, function_span
from toolathlon_native_trace import export_native_trace

async def main():
    async with export_native_trace():
        with generation_span(model='local-test') as first:
            first.span_data.input = [{'role': 'user', 'content': 'x' * 100000}]
            first.span_data.output = [{'role': 'assistant', 'tool_calls': [
                {'id': 'call-probe', 'type': 'function',
                 'function': {'name': 'probe', 'arguments': '{}'}}]}]
            first.span_data.usage = {'input_tokens': 10, 'output_tokens': 3}
            await asyncio.sleep(0.03)
        with function_span('probe') as tool:
            tool.span_data.input = '{}'
            tool.span_data.output = 'real tool output'
            await asyncio.sleep(0.02)
        if marker := os.getenv('PAUSE_BEFORE_FINISH'):
            Path(marker).write_text('ready')
            await asyncio.sleep(60)
        with generation_span(model='local-test') as second:
            second.span_data.input = [{'role': 'tool', 'content': 'real tool output'}]
            second.span_data.output = [{'role': 'assistant', 'content': 'done'}]
            second.span_data.usage = {'input_tokens': 90, 'output_tokens': 7}
            await asyncio.sleep(0.03)
        if os.getenv('FAIL_AFTER_SPANS'):
            raise RuntimeError('planned agent failure')

try:
    asyncio.run(main())
except RuntimeError as exc:
    assert str(exc) == 'planned agent failure'
"""


def _spans(capture):
    result = []
    for batch in parse_jsonl(Path(capture.receipt.trace_uri).read_bytes()):
        result.extend(
            span for resource in batch.resource_spans for scope in resource.scope_spans for span in scope.spans
        )
    return result


@pytest.mark.parametrize("fail", [False, True])
def test_real_sdk_spans_export_over_otlp_with_measured_usage_and_time(tmp_path: Path, fail: bool) -> None:
    with local_capture(tmp_path / "traces", case_id="native-case", candidate_id="native-candidate") as capture:
        with NativeTrace(capture.trace) as receiver:
            status = tmp_path / "status.json"
            receiver.status_path = status
            env = (
                os.environ
                | receiver.env()
                | {
                    "PYTHONPATH": str(INTEGRATION),
                    "TOOLATHLON_TELEMETRY_STATUS": str(status),
                    "OPENAI_AGENTS_DISABLE_TRACING": "1",
                }
            )
            if fail:
                env["FAIL_AFTER_SPANS"] = "1"
            child = subprocess.run(
                [sys.executable, "-c", NATIVE_CHILD], env=env, text=True, capture_output=True, timeout=30
            )
            assert child.returncode == 0, child.stderr
            receiver.read_status()
    assert not capture.receipt.omissions
    spans = _spans(capture)
    models = []
    tools = []
    root_context = capture.trace._root_span.get_span_context()
    for span in spans:
        attrs = {item.key: item.value for item in span.attributes}
        if "beaker.operation.kind" not in attrs:
            continue
        if attrs["beaker.operation.kind"].string_value == "model":
            models.append(span)
        if attrs["beaker.operation.kind"].string_value == "tool" and "gen_ai.tool.name" in attrs:
            tools.append(span)
        assert span.trace_id == root_context.trace_id.to_bytes(16, "big")
        assert attrs["beaker.case.id"].string_value == "native-case"
        assert attrs["beaker.candidate.id"].string_value == "native-candidate"
    assert len(models) == 2
    usage = [{item.key: item.value.int_value for item in span.attributes} for span in models]
    assert sorted(item["gen_ai.usage.input_tokens"] for item in usage) == [10, 90]
    assert sorted(item["gen_ai.usage.output_tokens"] for item in usage) == [3, 7]
    assert all(span.end_time_unix_nano - span.start_time_unix_nano >= 20_000_000 for span in models)
    assert len(tools) == 1
    assert any(
        item.key == "gen_ai.tool.call.id" and item.value.string_value == "call-probe" for item in tools[0].attributes
    )
    # Large input content is retained as a parent-owned artifact, not a dead
    # child-process reference or a truncated OTel attribute.
    assert "x" * 100000 in "".join(path.read_text() for path in (tmp_path / "traces/content").glob("*"))
    known = {span.span_id for span in spans}
    assert all(span.parent_span_id in known or span.parent_span_id == b"" for span in spans)


def test_missing_native_telemetry_is_reported_as_incomplete(tmp_path: Path) -> None:
    with local_capture(tmp_path, case_id="no-telemetry") as capture:
        with NativeTrace(capture.trace):
            pass
    assert capture.receipt.status == "incomplete"
    assert any("no native OTLP" in omission for omission in capture.receipt.omissions)


def test_wrong_case_export_token_is_rejected(tmp_path: Path) -> None:
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    with local_capture(tmp_path, case_id="token-case") as capture:
        with NativeTrace(capture.trace) as receiver:
            request = Request(
                receiver.env()["TOOLATHLON_OTLP_ENDPOINT"],
                data=b"",
                method="POST",
                headers={"Content-Type": "application/x-protobuf", "x-beaker-capture-token": "another-case"},
            )
            with pytest.raises(HTTPError) as error:
                urlopen(request, timeout=5)
            assert error.value.code == 403


@pytest.mark.skipif(sys.platform != "linux", reason="Production agent uses Linux SIGTERM")
def test_sigterm_flushes_native_spans_before_process_exits(tmp_path: Path) -> None:
    marker, status = tmp_path / "ready", tmp_path / "status.json"
    with local_capture(tmp_path / "trace", case_id="cancelled-case") as capture:
        with NativeTrace(capture.trace) as receiver:
            receiver.status_path = status
            env = (
                os.environ
                | receiver.env()
                | {
                    "PYTHONPATH": str(INTEGRATION),
                    "TOOLATHLON_TELEMETRY_STATUS": str(status),
                    "PAUSE_BEFORE_FINISH": str(marker),
                }
            )
            child = subprocess.Popen(
                [sys.executable, "-c", NATIVE_CHILD], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            try:
                deadline = time.monotonic() + 10
                while not marker.is_file() and time.monotonic() < deadline and child.poll() is None:
                    time.sleep(0.01)
                assert marker.is_file()
                child.send_signal(signal.SIGTERM)
                child.communicate(timeout=10)
                receiver.read_status()
            finally:
                if child.poll() is None:
                    child.kill()
                    child.communicate()
    assert status.is_file()
    assert not capture.receipt.omissions
    assert capture.receipt.counts["model_calls"] == 1
    assert any(
        item.key == "gen_ai.tool.name" and item.value.string_value == "probe"
        for span in _spans(capture)
        for item in span.attributes
    )
