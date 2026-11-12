"""Port Toolathlon's context hooks to the pinned OpenAI Agents 0.20 runtime.

Only the staged agent source is patched. Official assets, graders and their
original environment remain immutable. Replacements fail if upstream drifts.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def _replace(path: Path, before: str, after: str) -> None:
    text = path.read_text()
    if text.count(before) != 1:
        raise ValueError(f"Unexpected pinned upstream source in {path}: {before!r}")
    path.write_text(text.replace(before, after))


def patch_source(checkout: Path) -> None:
    provider = checkout / "utils/api_model/model_provider.py"
    _replace(provider, "set_tracing_disabled(disabled=True)", "# Tracing is owned by the Beaker launch adapter.")
    _replace(
        provider,
        "from openai import AsyncOpenAI",
        "from openai import AsyncOpenAI, NOT_GIVEN, NotGiven\nfrom typing import cast\n"
        "from openai.types.chat import (\n"
        "    ChatCompletionMessageFunctionToolCallParam as ChatCompletionMessageToolCallParam,\n)\n"
        "from agents.util._error_tracing import model_span_errors",
    )
    _replace(
        provider,
        "        ) as span_generation:",
        "        ) as span_generation, model_span_errors(\n"
        "            span_generation, message='Toolathlon model request failed',\n"
        "            trace_include_sensitive_data=tracing.include_data(),\n        ):",
    )
    _replace(
        provider,
        "        previous_response_id: str | None,\n    ) -> ModelResponse:",
        "        previous_response_id: str | None,\n"
        "        conversation_id: str | None = None,\n"
        "        prompt: Any = None,\n    ) -> ModelResponse:\n"
        "        if conversation_id is not None or prompt is not None:\n"
        "            raise ValueError('The Toolathlon unified provider uses local conversation state')",
    )
    _replace(
        provider,
        '                "output_tokens": usage.output_tokens,\n            }',
        '                "output_tokens": usage.output_tokens,\n'
        '                "input_tokens_details": {"cached_tokens": usage.cached_input_tokens},\n            }',
    )
    context = checkout / "utils/roles/context_managed_runner.py"
    _replace(context, "from agents._run_impl import AgentToolUseTracker, SingleStepResult", "")
    text = context.read_text()
    start = text.index("    @classmethod\n    async def _run_single_turn(")
    end = text.index("    @classmethod\n    def _handle_truncation(", start)
    context.write_text(text[:start] + text[end:])
    task_agent = checkout / "utils/roles/task_agent.py"
    _replace(task_agent, "from agents import (", "from agents import (\n    RunContextWrapper,")
    _replace(
        task_agent,
        "await self.agent.get_all_tools()",
        "await self.agent.get_all_tools(RunContextWrapper(self.shared_context))",
    )
    _replace(
        task_agent,
        "RunConfig(model_provider=self.agent_model_provider)",
        "RunConfig(model_provider=self.agent_model_provider, "
        "trace_include_sensitive_data=True, tracing_disabled=False)",
    )
    # Keep Toolathlon's exact overlong-output helper, without importing the
    # obsolete RunImpl patch which the new SDK no longer executes.
    old_patch = (checkout / "utils/openai_agents_monkey_patch/custom_run_impl.py").read_text()
    start = old_patch.index("def _truncate_overlong_tool_output(")
    end = old_patch.index("\n\n@classmethod", start)
    helper = (
        "import os, logging, shortuuid\nfrom typing import Any\nfrom agents import RunContextWrapper\n"
        "logger = logging.getLogger(__name__)\n"
        "MAX_SINGLE_TURN_RETURN_CHARS = int(os.getenv('BENCH_MAX_SINGLE_TURN_RETURN_CHARS', 100000))\n"
        + old_patch[start:end]
    )
    (checkout / "toolathlon_output_limit.py").write_text(helper)
    mcp = checkout / "utils/openai_agents_monkey_patch/custom_mcp_util.py"
    _replace(
        mcp,
        '    cls, tool: "MCPTool", server: "MCPServer", convert_schemas_to_strict: bool\n',
        '    cls, tool: "MCPTool", server: "MCPServer", convert_schemas_to_strict: bool, agent=None, **kwargs\n',
    )


_sdk_turn: Any = None


async def _context_turn(**kwargs: Any) -> Any:
    from utils.api_model.model_provider import ContextTooLongError
    from utils.roles.context_managed_runner import ContextManagedRunner

    context = kwargs["context_wrapper"]
    ctx = context.context
    agent = kwargs["bindings"].execution_agent
    meta = ctx["_context_meta"]
    for key in ("current_turn", "total_turns_ever", "turns_in_current_sequence"):
        meta[key] = meta.get(key, 0) + 1
    try:
        result = await _sdk_turn(**kwargs)
    except ContextTooLongError as exc:
        ctx["_force_reset_context"] = {
            "reason": str(exc),
            "token_count": exc.token_count,
            "max_tokens": exc.max_tokens,
            "timestamp": datetime.now().isoformat(),
            "executed_mini_turns": meta.get("mini_turns_in_current_sequence", 0),
            "executed_turns": meta["turns_in_current_sequence"],
        }
        raise
    start = meta["mini_turns_in_current_sequence"]
    meta["boundary_in_current_sequence"].append((start, start + len(result.new_step_items)))
    meta["mini_turns_in_current_sequence"] += len(result.new_step_items)
    ctx["_cumulative_usage"] = {
        name: getattr(context.usage, name) for name in ("total_tokens", "input_tokens", "output_tokens", "requests")
    }
    ContextManagedRunner._save_items_to_history(
        session_id=ctx["_session_id"],
        turn_number=meta["current_turn"],
        items=result.new_step_items,
        agent_name=agent.name,
        history_dir=Path(ctx["_history_dir"]),
    )
    if pending := ctx.get("_pending_truncate"):
        ContextManagedRunner._handle_truncation(
            original_input=result.original_input,
            pre_step_items=result.pre_step_items,
            new_step_items=result.new_step_items,
            truncate_params=pending,
            context_wrapper=context,
        )
        ctx["_pending_truncate"] = None
    return result


def install() -> None:
    """Install the context/history hook on the new SDK's actual execution path."""
    import agents.run as sdk_run
    from agents import OpenAIChatCompletionsModel, OpenAIResponsesModel
    from openai import NOT_GIVEN
    from utils.roles.task_agent import TaskAgent
    from utils.task_runner.hooks import RunLifecycle

    global _sdk_turn
    if _sdk_turn is not None:
        return
    _sdk_turn = sdk_run.run_single_turn
    sdk_run.run_single_turn = _context_turn
    for model in (OpenAIChatCompletionsModel, OpenAIResponsesModel):
        model._non_null_or_not_given = staticmethod(lambda value: NOT_GIVEN if value is None else value)

    async def on_llm_end(self: Any, context: Any, agent: Any, response: Any) -> None:
        context.context["_last_response_usage"] = {
            name: getattr(response.usage, name)
            for name in ("total_tokens", "input_tokens", "output_tokens", "requests")
        }

    RunLifecycle.on_llm_end = on_llm_end

    original_setup = TaskAgent.setup_agent

    async def setup_agent(self: Any) -> None:
        from toolathlon_output_limit import _truncate_overlong_tool_output

        await original_setup(self)
        for tool in self.agent.tools:
            invoke = tool.on_invoke_tool

            async def limited(context: Any, params: str, invoke: Any = invoke) -> Any:
                return _truncate_overlong_tool_output(await invoke(context, params), context)

            tool.on_invoke_tool = limited

    TaskAgent.setup_agent = setup_agent
