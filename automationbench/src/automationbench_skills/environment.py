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

from automationbench_skills.grading_state import capture_grading_state
from automationbench_skills.scenarios import ScenarioAPIContract
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

    def register(self, mcp: Any) -> None:
        super().register(mcp)
        if self._task.skills_dir:
            set_skills_dir(Path(self._task.skills_dir))
            for tool in SKILL_TOOLS:
                mcp.add_tool(tool, structured_output=False)

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
