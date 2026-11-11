import unittest

from claude_agent_sdk.types import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

from scripts.decoupled.host_agent_loop_claude_sdk import (
    build_allowed_mcp_tool_names,
    build_runtime_system_prompt,
    decide_task_status,
    get_env_optional_int,
    has_claude_sdk_auth,
    is_claude_code_builtin_model_name,
    parse_assistant_message,
    parse_user_message_content,
    resolve_claude_sdk_model,
    resolve_claude_sdk_env,
)
from utils.roles.task_agent import TaskStatus


class HostAgentLoopClaudeSDKTests(unittest.TestCase):
    def test_build_allowed_mcp_tool_names(self) -> None:
        names = build_allowed_mcp_tool_names(
            gateway_server_name="gw",
            tool_names=["fetch_json", "local-claim_done", "fetch_json"],
        )
        self.assertEqual(
            names,
            [
                "mcp__gw__fetch_json",
                "mcp__gw__local-claim_done",
            ],
        )

    def test_resolve_claude_sdk_env_maps_auth_token(self) -> None:
        resolved = resolve_claude_sdk_env(
            {
                "ANTHROPIC_BASE_URL": "https://api.example.com",
                "ANTHROPIC_AUTH_TOKEN": "token-a",
            }
        )
        self.assertEqual(resolved["ANTHROPIC_BASE_URL"], "https://api.example.com")
        self.assertEqual(resolved["ANTHROPIC_AUTH_TOKEN"], "token-a")

    def test_resolve_claude_sdk_env_prefers_api_key(self) -> None:
        resolved = resolve_claude_sdk_env(
            {
                "ANTHROPIC_API_KEY": "token-api",
                "ANTHROPIC_AUTH_TOKEN": "token-auth",
            }
        )
        self.assertEqual(resolved["ANTHROPIC_API_KEY"], "token-api")
        self.assertEqual(resolved["ANTHROPIC_AUTH_TOKEN"], "token-auth")

    def test_resolve_claude_sdk_env_preserves_empty_api_key(self) -> None:
        resolved = resolve_claude_sdk_env(
            {
                "ANTHROPIC_API_KEY": "",
                "ANTHROPIC_AUTH_TOKEN": "token-auth",
            }
        )
        self.assertIn("ANTHROPIC_API_KEY", resolved)
        self.assertEqual(resolved["ANTHROPIC_API_KEY"], "")
        self.assertEqual(resolved["ANTHROPIC_AUTH_TOKEN"], "token-auth")

    def test_resolve_claude_sdk_env_ignores_toolathlon_openai_env(self) -> None:
        resolved = resolve_claude_sdk_env(
            {
                "TOOLATHLON_OPENAI_BASE_URL": "https://ignored.example.com",
                "TOOLATHLON_OPENAI_API_KEY": "ignored-token",
            }
        )
        self.assertEqual(resolved, {})

    def test_resolve_claude_sdk_env_prefers_overrides(self) -> None:
        resolved = resolve_claude_sdk_env(
            {
                "ANTHROPIC_BASE_URL": "https://old.example.com",
                "ANTHROPIC_API_KEY": "old-token",
            },
            base_url_override="https://new.example.com",
            api_key_override="new-token",
        )
        self.assertEqual(resolved["ANTHROPIC_BASE_URL"], "https://new.example.com")
        self.assertEqual(resolved["ANTHROPIC_API_KEY"], "new-token")

    def test_is_claude_code_builtin_model_name(self) -> None:
        self.assertTrue(is_claude_code_builtin_model_name("sonnet"))
        self.assertTrue(is_claude_code_builtin_model_name("claude-sonnet-4-6"))
        self.assertTrue(
            is_claude_code_builtin_model_name("anthropic.claude-sonnet-4-5")
        )
        self.assertFalse(is_claude_code_builtin_model_name("google/gemini-3-flash-preview"))

    def test_resolve_claude_sdk_model_maps_custom_model_to_sonnet(self) -> None:
        cli_model, env_updates, requested = resolve_claude_sdk_model(
            "google/gemini-3-flash-preview",
        )
        self.assertEqual(cli_model, "sonnet")
        self.assertEqual(
            env_updates,
            {"ANTHROPIC_DEFAULT_SONNET_MODEL": "google/gemini-3-flash-preview"},
        )
        self.assertEqual(requested, "google/gemini-3-flash-preview")

    def test_resolve_claude_sdk_model_keeps_builtin_model(self) -> None:
        cli_model, env_updates, requested = resolve_claude_sdk_model("claude-sonnet-4-6")
        self.assertEqual(cli_model, "claude-sonnet-4-6")
        self.assertEqual(env_updates, {})
        self.assertIsNone(requested)

    def test_has_claude_sdk_auth_accepts_auth_token_only(self) -> None:
        self.assertTrue(has_claude_sdk_auth({"ANTHROPIC_AUTH_TOKEN": "token"}))
        self.assertFalse(has_claude_sdk_auth({"ANTHROPIC_API_KEY": ""}))

    def test_get_env_optional_int_returns_none_for_empty(self) -> None:
        self.assertIsNone(get_env_optional_int("NOT_SET_INT_ENV"))

    def test_parse_assistant_message_extracts_tool_calls(self) -> None:
        message = AssistantMessage(
            content=[
                TextBlock(text="hello"),
                ToolUseBlock(id="call_1", name="search", input={"q": "toolathlon"}),
            ],
            model="claude-sonnet-4-6",
        )

        entry, tool_calls, no_tool = parse_assistant_message(message)

        self.assertEqual(entry["role"], "assistant")
        self.assertEqual(entry["content"], "hello")
        self.assertEqual(entry["tool_calls_count"], 1)
        self.assertFalse(no_tool)
        self.assertEqual(tool_calls[0]["function"]["name"], "search")

    def test_parse_assistant_message_no_tool_call(self) -> None:
        message = AssistantMessage(
            content=[TextBlock(text="final answer")],
            model="claude-sonnet-4-6",
        )

        entry, tool_calls, no_tool = parse_assistant_message(message)

        self.assertEqual(entry["tool_calls_count"], 0)
        self.assertEqual(tool_calls, [])
        self.assertTrue(no_tool)

    def test_parse_user_message_content_blocks(self) -> None:
        text, blocks = parse_user_message_content(
            [
                ToolResultBlock(tool_use_id="call_1", content="done", is_error=False),
                TextBlock(text="ack"),
            ]
        )
        self.assertEqual(text, "done\nack")
        self.assertIsNotNone(blocks)
        assert blocks is not None
        self.assertEqual(blocks[0]["type"], "tool_result")

    def test_decide_task_status_prefers_no_tool_call(self) -> None:
        result_message = ResultMessage(
            subtype="success",
            duration_ms=100,
            duration_api_ms=80,
            is_error=False,
            num_turns=2,
            session_id="sess_1",
            result="ok",
        )

        status, reason = decide_task_status(result_message, saw_stop_tool=True, saw_no_tool_call_turn=True)
        self.assertEqual(status, TaskStatus.SUCCESS)
        self.assertEqual(reason, "no_tool_call")

    def test_decide_task_status_max_turns(self) -> None:
        result_message = ResultMessage(
            subtype="error",
            duration_ms=100,
            duration_api_ms=80,
            is_error=True,
            num_turns=3,
            session_id="sess_2",
            result="Maximum turns reached",
        )

        status, reason = decide_task_status(result_message, saw_stop_tool=False, saw_no_tool_call_turn=False)
        self.assertEqual(status, TaskStatus.MAX_TURNS_REACHED)
        self.assertEqual(reason, "max_turns_reached")

    def test_build_runtime_system_prompt_contains_serial_tool_constraints(self) -> None:
        prompt = build_runtime_system_prompt("base prompt", "serial")
        self.assertIn("base prompt", prompt)
        self.assertIn("At most one tool call per assistant turn", prompt)
        self.assertIn("Do not dispatch parallel sibling tool calls", prompt)

    def test_build_runtime_system_prompt_parallel_mode_no_serial_guard(self) -> None:
        prompt = build_runtime_system_prompt("base prompt", "parallel")
        self.assertEqual(prompt, "base prompt")


if __name__ == "__main__":
    unittest.main()
