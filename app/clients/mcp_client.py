import asyncio
from typing import Any

from agents.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams
from mcp.types import CallToolResult, TextContent

from app.conf.bailian_mcp_config import mcp_config
from app.core.logger import logger


class McpClient:
    """基于 OpenAI Agents SDK 的百炼 MCP 客户端。"""

    def __init__(self):
        self.url = mcp_config.mcp_base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {mcp_config.api_key}",
        }

    def call_tool(self, tool_name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._call_tool_async(tool_name, args or {}))
        finally:
            loop.close()

    async def _call_tool_async(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        params = MCPServerStreamableHttpParams(url=self.url, headers=self.headers)
        server = MCPServerStreamableHttp(params=params)
        try:
            await server.connect()
            result: CallToolResult = await server.call_tool(tool_name, args)
            return {
                "content": [
                    {"type": c.type, "text": c.text}
                    for c in result.content
                    if isinstance(c, TextContent)
                ],
                "isError": result.isError,
            }
        except Exception as e:
            logger.error(f"MCP 工具调用失败 [{tool_name}]: {e}")
            return {"content": [], "isError": True}
        finally:
            try:
                await server.disconnect()
            except Exception:
                pass


_mcp_client: McpClient | None = None


def get_mcp_client() -> McpClient:
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = McpClient()
    return _mcp_client
