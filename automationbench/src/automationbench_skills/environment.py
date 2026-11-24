"""Small Beaker extensions to PI's native task and MCP toolset."""

from __future__ import annotations

import contextvars
import json
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from automationbench.runner import (
    AutomationBenchData,
    AutomationBenchTask,
    AutomationBenchTaskConfig,
    AutomationBenchToolset,
)
from beaker.tracing import current_trace
from mcp.types import CallToolResult

from automationbench_skills.api_tools import (
    explain_confluence_fields,
    explain_salesforce_fields,
    expose_contact_phone_fields,
    prefer_requested_endpoints,
)
from automationbench_skills.gmail_plaintext import (
    expose_gmail_plaintext,
    gmail_fetch_schema,
    plaintext_error,
    prepare_gmail_body,
)
from automationbench_skills.grading_state import capture_grading_state
from automationbench_skills.scenarios import ScenarioAPIContract
from automationbench_skills.salesforce_context import SalesforceQueryContext
from automationbench_skills.sheets_tools import make_sheets_update_rows
from automationbench_skills.skills_tools import SKILL_TOOLS, set_skills_dir


class BeakerData(AutomationBenchData):
    skills_dir: str | None = None
    scenario: dict[str, Any] = {}


class BeakerToolset(AutomationBenchToolset):
    TOOL_PREFIX = None

    async def setup_task(self, task: BeakerData) -> None:
        await super().setup_task(task)
        if fault := task.scenario.get("tool_fault"):
            if self.config.mode != "api":
                raise ValueError("Pilot fault fixtures require the API toolset")
            self._api = ScenarioAPIContract(self._world, fault)
        if self.config.mode == "api":
            expose_contact_phone_fields(self._api)
            explain_salesforce_fields(self._api)
            explain_confluence_fields(self._api)
            self._salesforce_context = SalesforceQueryContext(self._api)
            expose_gmail_plaintext(self._api)
            prefer_requested_endpoints(self._api)

    def register(self, mcp: Any) -> None:
        super().register(mcp)
        if self.config.mode == "api":
            fetch_tool = mcp._tool_manager.get_tool("api_fetch")
            fetch_tool.parameters = gmail_fetch_schema(self._api.fetch_schema())
        if self.config.mode == "api" and {
            "sheets.spreadsheets.get", "sheets.spreadsheets.values.update"
        }.issubset(self._api.endpoints):
            mcp.add_tool(
                self._serialized(self._with_state(make_sheets_update_rows(self._make_api_fetch()))),
                name="sheets_update_rows",
                structured_output=False,
            )
        if self._task.skills_dir:
            set_skills_dir(Path(self._task.skills_dir))
            for tool in SKILL_TOOLS:
                mcp.add_tool(tool, structured_output=False)

    def _make_api_fetch(self):
        native = super()._make_api_fetch()

        def call(
            endpoint: str,
            path: dict[str, Any] | None = None,
            params: dict[str, Any] | None = None,
            body: dict[str, Any] | None = None,
        ) -> CallToolResult:
            if endpoint in self._api.endpoints:
                try:
                    body = prepare_gmail_body(endpoint, body)
                except ValueError as error:
                    self._record_partial_credit()
                    return plaintext_error(endpoint, error)
            result = native(endpoint, path, params, body)
            return self._salesforce_context.enrich(endpoint, params, result)

        call.__name__ = native.__name__
        call.__doc__ = native.__doc__
        return call

    def _record_partial_credit(self) -> None:
        super()._record_partial_credit()
        # PI shares state with the host after each write. Export the evidence
        # there too, so timeout/failure never discards completed agent work.
        self.state.artifacts["world.json"] = self._world.model_dump_json().encode()
        self.state.artifacts["grading-state.json"] = json.dumps(capture_grading_state(self._world)).encode()


class BeakerTask(AutomationBenchTask):
    @classmethod
    def data_type(cls) -> type[BeakerData]:
        return BeakerData

    @classmethod
    def toolsets(cls, config: AutomationBenchTaskConfig) -> list[vf.Toolset]:
        return [BeakerToolset(config.tools)]

    def __init__(self, data: BeakerData, config: AutomationBenchTaskConfig) -> None:
        super().__init__(data, config)
        self._tool_spans: dict[str, tuple[Any, Any]] = {}
        self._finished: set[str] = set()

    @vf.intercept
    def trace_tools(self, request: vf.Request) -> None:
        # PI calls this hook with an empty result before tool execution, then
        # with the actual result before the next model turn.
        message = request.messages[-1]
        if not isinstance(message, vf.ToolMessage) or message.tool_call_id in self._finished:
            return
        identifier = message.tool_call_id
        if identifier not in self._tool_spans:
            calls = [c for m in request.messages if isinstance(m, vf.AssistantMessage) for c in m.tool_calls or []]
            call = next(c for c in calls if c.id == identifier)
            if self.data.scenario.get("trace_capture") == "incomplete" and len(self._finished) % 2 == 0:
                self._finished.add(identifier)
                return
            context = contextvars.copy_context()
            try:
                arguments = json.loads(call.arguments)
            except json.JSONDecodeError:
                arguments = call.arguments
            operation = current_trace().tool_call(call.name, arguments=arguments, call_id=identifier)
            context.run(operation.__enter__)
            self._tool_spans[identifier] = (context, operation)
        else:
            context, operation = self._tool_spans.pop(identifier)
            operation.output(message.content)
            context.run(operation.__exit__, None, None, None)
            self._finished.add(identifier)

    def close_spans(self) -> None:
        for context, operation in self._tool_spans.values():
            error = RuntimeError("Tool result was not captured before the rollout ended")
            context.run(operation.__exit__, type(error), error, None)
        self._tool_spans.clear()


if __name__ == "__main__":
    BeakerToolset.run()
