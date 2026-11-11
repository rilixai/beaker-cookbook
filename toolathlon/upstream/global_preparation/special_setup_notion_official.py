from utils.mcp.tool_servers import MCPServerManager, call_tool_with_retry, ToolCallError
import asyncio
import json
from pathlib import Path

# Ensure the MCP auth directory exists (will be used by mcp-remote via env var in yaml config)
repo_root = Path(__file__).parent.parent.resolve()
mcp_auth_dir = repo_root / "configs" / ".mcp-auth"
mcp_auth_dir.mkdir(parents=True, exist_ok=True)

xx_MCPServerManager = MCPServerManager(agent_workspace="./") # a pseudo server manager
notion_official_server = xx_MCPServerManager.servers['notion_official']

from configs.token_key_session import all_token_key_session

async def main():
    print(f"MCP authentication will be stored in: {mcp_auth_dir}")
    async with notion_official_server as server:
        print("We need to configure this notion official mcp server to the desires account, so that it can be used to duplicate and move pages!")
        print("Please follow the login guidances to do so ...")
        pass
    print(">>>> DONE!")

if __name__ == "__main__":
    asyncio.run(main())