# Implementation: MCP_Client

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::Doc|MCP Protocol|https://modelcontextprotocol.io]]
|-
! Domains
| [[domain::MCP]], [[domain::Browser_Automation]], [[domain::Protocol]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:00 GMT]]
|}

== Overview ==
Async context manager for Model Context Protocol (MCP) client that establishes connections to browser servers for tool execution.

=== Description ===
The `mcp_client.py` module provides an async context manager that connects to MCP-compatible browser automation servers. It handles:

- Server connection establishment and teardown
- Tool schema retrieval from MCP server
- Message passing for tool execution
- Connection lifecycle management

The MCP protocol enables standardized communication between LLM agents and external tools like browser automation.

=== Usage ===
Use `async with mcp_client()` to establish a connection to the MCP browser server before executing browser tools.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/NestBrowse/toolkit/mcp_client.py WebAgent/NestBrowse/toolkit/mcp_client.py]
* '''Lines:''' 1-49

=== Signature ===
<syntaxhighlight lang="python">
@asynccontextmanager
async def mcp_client(
    server_url: str = "ws://localhost:8765"
) -> AsyncGenerator[MCPClient, None]:
    """
    Async context manager for MCP client connection.

    Args:
        server_url: WebSocket URL of MCP server

    Yields:
        MCPClient: Connected client instance

    Example:
        async with mcp_client() as client:
            tools = client.get_tools()
            result = await client.call_tool("visit", {"url": "..."})
    """
    ...

class MCPClient:
    """MCP protocol client for browser tool execution."""

    def get_tools(self) -> List[Dict]:
        """Get available tool schemas from server."""
        ...

    async def call_tool(
        self,
        tool_name: str,
        params: Dict
    ) -> str:
        """Execute a tool on the MCP server."""
        ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from WebAgent.NestBrowse.toolkit.mcp_client import mcp_client
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| server_url || str || No || MCP server WebSocket URL (default ws://localhost:8765)
|-
| tool_name || str || Yes || Name of tool to execute
|-
| params || Dict || Yes || Tool parameters
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| client || MCPClient || Connected MCP client
|-
| tools || List[Dict] || Available tool schemas
|-
| result || str || Tool execution result
|}

== Usage Examples ==

=== Basic MCP Connection ===
<syntaxhighlight lang="python">
import asyncio
from WebAgent.NestBrowse.toolkit.mcp_client import mcp_client

async def browse_with_mcp():
    async with mcp_client() as client:
        # Get available tools
        tools = client.get_tools()
        print(f"Available tools: {[t['name'] for t in tools]}")

        # Execute visit tool
        result = await client.call_tool(
            "visit",
            {"url": "https://example.com"}
        )
        print(result)

asyncio.run(browse_with_mcp())
</syntaxhighlight>

=== Custom Server URL ===
<syntaxhighlight lang="python">
async with mcp_client(server_url="ws://browser-server:9000") as client:
    # Use custom MCP server
    result = await client.call_tool("click", {"selector": "button"})
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_MCP_Protocol]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_Python_Dependencies]]
