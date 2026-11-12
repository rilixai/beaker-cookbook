"""Export real Agents SDK spans from the child and adopt OTLP in its Beaker case."""

from __future__ import annotations

import asyncio
import json
import os
import signal
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from beaker.tracing.core import CaptureTrace, NoopTrace, activate_trace, capture_identity, content_reference
from beaker.tracing.otlp_receiver import TOKEN_HEADER, OtlpReceiver


def _set(attributes: Any, name: str, value: str) -> None:
    for item in attributes:
        if item.key == name:
            item.value.string_value = value
            return
    attributes.add(key=name).value.string_value = value


class NativeTrace:
    """Only the parent holds run credentials; the child gets a per-case token."""

    def __init__(self, trace: Any) -> None:
        self.trace = trace
        self.receiver = OtlpReceiver() if isinstance(trace, CaptureTrace) and trace.enabled else None
        self.status_path: Path | None = None
        self.status: dict[str, Any] | None = None

    def __enter__(self) -> NativeTrace:
        return self

    def env(self) -> dict[str, str]:
        endpoint = self.receiver.endpoint if self.receiver else None
        return {
            "TOOLATHLON_OTLP_ENDPOINT": endpoint.traces_url if endpoint else "",
            "TOOLATHLON_OTLP_TOKEN": endpoint.token if endpoint else "",
        }

    def read_status(self) -> None:
        # The per-case directory is removed before this outer context exits.
        if self.status_path is not None and self.status_path.is_file():
            try:
                self.status = json.loads(self.status_path.read_text())
            except (ValueError, OSError) as exc:
                self.status = {"omissions": [f"Invalid native telemetry status: {exc}"]}

    def __exit__(self, *args: Any) -> None:
        if self.receiver is None:
            return
        receiver = self.receiver
        session = self.trace.session
        try:
            batches, _ = receiver.collected()
            spans = [
                span
                for batch in batches
                for resource in batch.resource_spans
                for scope in resource.scope_spans
                for span in scope.spans
            ]
            root = self.trace._root_span.get_span_context()
            known = {span.span_id for span in spans}
            identity = capture_identity(self.trace)
            for span in spans:
                # Keep measured start/end times and the child's real hierarchy.
                span.trace_id = root.trace_id.to_bytes(16, "big")
                if span.parent_span_id not in known:
                    span.parent_span_id = root.span_id.to_bytes(8, "big")
                for key, value in identity.items():
                    _set(span.attributes, key, value)
                for item in span.attributes:
                    if item.key not in {
                        "input.value",
                        "output.value",
                        "gen_ai.tool.call.arguments",
                        "gen_ai.tool.call.result",
                    }:
                        continue
                    raw = item.value.string_value
                    try:
                        data = json.loads(raw)
                    except (ValueError, TypeError):
                        data = raw
                    referenced = content_reference(self.trace, item.key, data)
                    item.value.string_value = referenced if isinstance(referenced, str) else json.dumps(referenced)
            for batch in batches:
                for resource in batch.resource_spans:
                    for key, value in identity.items():
                        _set(resource.resource.attributes, key, value)
                session.manager.adopt(session, batch)
            if not spans:
                session.omissions.append("Toolathlon agent exported no native OTLP spans")
            if receiver.rejected or receiver.dropped:
                session.omissions.append(
                    f"Toolathlon OTLP batches rejected={receiver.rejected}, dropped={receiver.dropped}"
                )
            if self.status is None:
                session.omissions.append("Toolathlon agent did not confirm telemetry flush")
            else:
                session.omissions.extend(str(item) for item in self.status.get("omissions", []))
        except Exception as exc:
            session.omissions.append(f"Toolathlon native telemetry adoption failed: {exc}")
        finally:
            receiver.close()


@asynccontextmanager
async def export_native_trace():
    """Use Beaker's real framework connector, not a replay of traj_log.json."""
    from agents import set_trace_processors, set_tracing_disabled
    from agents import trace as agents_trace
    from beaker.tracing.integrations import openai_agents
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExportResult

    # This dedicated agent process exports only to the parent case. Removing
    # Toolathlon's disabling statement alone would enable the OpenAI exporter.
    os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "0"
    set_trace_processors([])
    set_tracing_disabled(False)
    endpoint = os.environ.get("TOOLATHLON_OTLP_ENDPOINT")
    if not endpoint:
        yield
        return

    omissions: list[str] = []

    class Exporter(OTLPSpanExporter):
        def export(self, spans: Any) -> Any:
            result = super().export(spans)
            if result != SpanExportResult.SUCCESS:
                omissions.append("Toolathlon native OTLP export failed")
            return result

    provider = TracerProvider()
    provider.add_span_processor(
        BatchSpanProcessor(
            Exporter(
                endpoint=endpoint,
                headers={TOKEN_HEADER: os.environ["TOOLATHLON_OTLP_TOKEN"]},
                timeout=5,
            ),
            schedule_delay_millis=200,
        )
    )

    class ExportTrace(NoopTrace):
        enabled = True
        tracer_provider = provider
        session = SimpleNamespace(omissions=omissions)
        _root_span = provider.get_tracer("toolathlon.agent", "0.20.0").start_span(
            "toolathlon.agent", attributes={"beaker.operation.kind": "stage"}
        )

    remote = ExportTrace()
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    if task is not None:
        loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        with activate_trace(remote), openai_agents.registered(remote):
            try:
                with agents_trace("Toolathlon"):
                    yield
            finally:
                await openai_agents.flush()
    finally:
        remote._root_span.end()
        if not provider.force_flush(timeout_millis=5000):
            omissions.append("Toolathlon native OTLP flush timed out")
        provider.shutdown()
        Path(os.environ["TOOLATHLON_TELEMETRY_STATUS"]).write_text(json.dumps({"omissions": omissions}))
        loop.remove_signal_handler(signal.SIGTERM)
