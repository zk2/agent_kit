"""Load the MCP tools as LangChain tools for the agent.

Uses the stdio transport: the adapter launches `python -m agentkit.tools.server` as a subprocess
and exposes its tools. Returning plain LangChain tools lets the graph bind them to the LLM and
execute them with a standard ToolNode.
"""

from __future__ import annotations

import sys

# Server identifier under which the tools are registered with the MCP client.
_SERVER_NAME = "agentkit"


def _connections() -> dict:
    return {
        _SERVER_NAME: {
            "command": sys.executable,
            "args": ["-m", "agentkit.tools.server"],
            "transport": "stdio",
        }
    }


async def load_mcp_tools() -> list:
    """Connect to the MCP server and return its tools as LangChain tools."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(_connections())
    return await client.get_tools()
