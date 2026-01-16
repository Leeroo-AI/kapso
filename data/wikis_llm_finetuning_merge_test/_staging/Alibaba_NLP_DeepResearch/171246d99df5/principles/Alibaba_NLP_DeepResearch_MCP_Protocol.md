# Principle: MCP_Protocol

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Model Context Protocol|https://modelcontextprotocol.io/]]
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Protocol]], [[domain::Agent_Systems]], [[domain::Tool_Execution]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:30 GMT]]
|}

== Overview ==

Model Context Protocol (MCP) integration for standardized communication between LLM agents and external tool servers, enabling modular browser automation.

=== Description ===

MCP Protocol is a standardized interface for connecting LLM applications to external capabilities. In the DeepResearch context, it enables:

1. **Server discovery** - Agents can query available tools from MCP servers
2. **Schema retrieval** - Get tool definitions in standard format
3. **Tool execution** - Call tools with parameters, receive results
4. **Async communication** - WebSocket-based message passing

This decouples the agent logic from tool implementation, allowing browser automation to run as a separate service.

=== Usage ===

Use MCP Protocol when:
- Building modular agent architectures
- Separating tool execution from agent logic
- Need to share tools across multiple agents
- Implementing browser automation services

== Theoretical Basis ==

MCP follows a client-server architecture:

'''MCP Communication Pattern:'''
<syntaxhighlight lang="python">
from contextlib import asynccontextmanager

@asynccontextmanager
async def mcp_client(server_url: str):
    """Connect to MCP server."""
    # Establish WebSocket connection
    connection = await websocket.connect(server_url)

    try:
        # Create client wrapper
        client = MCPClient(connection)

        # Discover available tools
        await client.initialize()

        yield client
    finally:
        await connection.close()

class MCPClient:
    def __init__(self, connection):
        self.connection = connection
        self.tools = {}

    async def initialize(self):
        """Get tool schemas from server."""
        response = await self.send("tools/list")
        self.tools = {t['name']: t for t in response['tools']}

    def get_tools(self) -> list:
        """Return tool definitions for LLM."""
        return list(self.tools.values())

    async def call_tool(self, name: str, params: dict) -> str:
        """Execute tool on server."""
        response = await self.send("tools/call", {
            "name": name,
            "arguments": params
        })
        return response['content']
</syntaxhighlight>

Key MCP concepts:
- **Tools**: Functions the server exposes
- **Resources**: Data the server provides
- **Prompts**: Templates the server offers
- **Sampling**: Server can request LLM generations

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_MCP_Client]]

=== Related Principles ===
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Browser_Agent]]
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Browser_Interaction]]
