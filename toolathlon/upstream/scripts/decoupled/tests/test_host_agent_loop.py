import os
import tempfile
import unittest

import yaml
from agents.items import MessageOutputItem, ToolCallItem, ToolCallOutputItem
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_refusal import ResponseOutputRefusal
from openai.types.responses.response_output_text import ResponseOutputText

from scripts.decoupled.host_agent_loop import (
    build_gateway_runtime_mcp_config,
    decoupled_termination_checker,
    expand_stop_tool_names,
    extract_assistant_console_entries,
    filter_local_tools,
    render_run_items_to_console_events,
)


class HostAgentLoopHelperTests(unittest.TestCase):
    def test_filter_local_tools_removes_ignored(self) -> None:
        tools = [
            "claim_done",
            "manage_context",
            "history",
            "handle_overlong_tool_outputs",
            "python_execute",
            "web_search",
        ]
        filtered = filter_local_tools(tools)
        self.assertEqual(filtered, ["python_execute", "web_search"])

    def test_expand_stop_tool_names_adds_gateway_prefix(self) -> None:
        expanded = expand_stop_tool_names(["local-claim_done", "local-custom"], "gw")
        self.assertIn("local-claim_done", expanded)
        self.assertIn("gw-local-claim_done", expanded)
        self.assertIn("local-custom", expanded)
        self.assertIn("gw-local-custom", expanded)

    def test_build_gateway_runtime_mcp_config_writes_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_dir = build_gateway_runtime_mcp_config(
                runtime_dir=tmpdir,
                gateway_server_name="gateway_a",
                gateway_url="http://127.0.0.1:10086/sse",
            )
            config_path = os.path.join(runtime_dir, "gateway_sse.yaml")
            self.assertTrue(os.path.exists(config_path))

            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["type"], "sse")
            self.assertEqual(config["name"], "gateway_a")
            self.assertEqual(config["params"]["url"], "http://127.0.0.1:10086/sse")

    def test_decoupled_termination_checker_stops_on_no_tool_call(self) -> None:
        should_stop = decoupled_termination_checker(
            content="final answer",
            recent_tools=[],
            check_target="agent",
            user_stop_phrases=["#### STOP"],
            agent_stop_tools=["local-claim_done"],
        )
        self.assertTrue(should_stop)

    def test_decoupled_termination_checker_stops_on_stop_tool(self) -> None:
        should_stop = decoupled_termination_checker(
            content="",
            recent_tools=[{"function": {"name": "local-claim_done"}}],
            check_target="agent",
            user_stop_phrases=[],
            agent_stop_tools=["local-claim_done"],
        )
        self.assertTrue(should_stop)

    def test_decoupled_termination_checker_user_phrase(self) -> None:
        should_stop = decoupled_termination_checker(
            content="please #### STOP now",
            recent_tools=[],
            check_target="user",
            user_stop_phrases=["#### STOP"],
            agent_stop_tools=[],
        )
        self.assertTrue(should_stop)

    def test_extract_assistant_console_entries_reads_text_and_refusal(self) -> None:
        raw_message = ResponseOutputMessage(
            id="msg_1",
            content=[
                ResponseOutputText(text="hello", type="output_text", annotations=[]),
                ResponseOutputRefusal(refusal="cannot do that", type="refusal"),
            ],
            role="assistant",
            status="completed",
            type="message",
        )

        entries = extract_assistant_console_entries(raw_message)

        self.assertEqual(entries[0][0], "ASSIST")
        self.assertEqual(entries[0][1], "hello")
        self.assertEqual(entries[1][0], "ASSIST")
        self.assertEqual(entries[1][1], "cannot do that")

    def test_render_run_items_to_console_events_renders_tool_sequence(self) -> None:
        items = [
            MessageOutputItem(
                agent=None,
                raw_item=ResponseOutputMessage(
                    id="msg_1",
                    content=[
                        ResponseOutputText(text="Looking it up", type="output_text", annotations=[])
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                ),
            ),
            ToolCallItem(
                agent=None,
                raw_item=ResponseFunctionToolCall(
                    arguments='{"keyword":"Alita"}',
                    call_id="call_1",
                    name="search_arxiv",
                    type="function_call",
                ),
            ),
            ToolCallOutputItem(
                agent=None,
                raw_item={
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "found paper",
                },
                output="found paper",
            ),
        ]

        events = render_run_items_to_console_events(items)

        self.assertEqual(
            [event[0] for event in events],
            ["ASSIST", "TOOL CALL", "TOOL OUT"],
        )
        self.assertIn("Looking it up", events[0][1])
        self.assertIn("search_arxiv#call_1", events[1][1])
        self.assertIn('{"keyword":"Alita"}', events[1][1])
        self.assertIn("search_arxiv#call_1 found paper", events[2][1])


if __name__ == "__main__":
    unittest.main()
