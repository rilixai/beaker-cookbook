import unittest

from scripts.decoupled.container_tool_gateway import GatewayCore, ToolRegistry


class ToolRegistryTests(unittest.TestCase):
    def test_registry_always_prefixes_remote_tool_names(self) -> None:
        registry = ToolRegistry()
        registry.add_remote_tools(
            "server_a",
            [
                {
                    "name": "search",
                    "description": "search a",
                    "inputSchema": {"type": "object", "properties": {}},
                }
            ],
        )
        registry.add_remote_tools(
            "server_b",
            [
                {
                    "name": "search",
                    "description": "search b",
                    "inputSchema": {"type": "object", "properties": {}},
                }
            ],
        )

        tools = registry.list_tools()
        names = [tool["name"] for tool in tools]
        self.assertIn("server_a-search", names)
        self.assertIn("server_b-search", names)


class GatewayCoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_initialize_and_tools_list(self) -> None:
        registry = ToolRegistry()
        registry.add_claim_done()
        core = GatewayCore(registry=registry, remote_caller=self._unreachable_remote)

        init_resp = await core.handle_json_rpc(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        )
        self.assertEqual(init_resp["id"], 1)
        self.assertIn("protocolVersion", init_resp["result"])

        list_resp = await core.handle_json_rpc(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        )
        names = [tool["name"] for tool in list_resp["result"]["tools"]]
        self.assertIn("local-claim_done", names)

    async def test_local_claim_done_call(self) -> None:
        registry = ToolRegistry()
        registry.add_claim_done()
        core = GatewayCore(registry=registry, remote_caller=self._unreachable_remote)

        call_resp = await core.handle_json_rpc(
            {
                "jsonrpc": "2.0",
                "id": "abc",
                "method": "tools/call",
                "params": {"name": "local-claim_done", "arguments": {}},
            }
        )
        self.assertEqual(call_resp["id"], "abc")
        self.assertFalse(call_resp["result"]["isError"])
        self.assertIn("claimed", call_resp["result"]["content"][0]["text"])

    async def test_remote_tool_call_delegates_to_remote_caller(self) -> None:
        registry = ToolRegistry()
        registry.add_remote_tools(
            "server_a",
            [
                {
                    "name": "search",
                    "description": "search",
                    "inputSchema": {"type": "object", "properties": {}},
                }
            ],
        )
        seen = {}

        async def remote_caller(record, arguments):
            seen["tool"] = record.backend_name
            seen["server"] = record.server_name
            seen["arguments"] = arguments
            return {
                "content": [{"type": "text", "text": "ok"}],
                "isError": False,
            }

        core = GatewayCore(registry=registry, remote_caller=remote_caller)
        call_resp = await core.handle_json_rpc(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "server_a-search", "arguments": {"q": "toolathlon"}},
            }
        )

        self.assertEqual(seen["tool"], "search")
        self.assertEqual(seen["server"], "server_a")
        self.assertEqual(seen["arguments"], {"q": "toolathlon"})
        self.assertEqual(call_resp["result"]["content"][0]["text"], "ok")

    async def test_unknown_method_returns_error(self) -> None:
        registry = ToolRegistry()
        core = GatewayCore(registry=registry, remote_caller=self._unreachable_remote)
        response = await core.handle_json_rpc(
            {"jsonrpc": "2.0", "id": 10, "method": "unknown"}
        )
        self.assertEqual(response["error"]["code"], -32601)

    async def _unreachable_remote(self, record, arguments):
        raise RuntimeError("should not be called")


if __name__ == "__main__":
    unittest.main()
