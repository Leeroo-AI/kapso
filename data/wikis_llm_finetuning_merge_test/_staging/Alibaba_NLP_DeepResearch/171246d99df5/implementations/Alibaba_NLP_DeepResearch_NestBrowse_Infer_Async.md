# Implementation: NestBrowse_Infer_Async

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Browser_Automation]], [[domain::Async_Agent]], [[domain::MCP]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:00 GMT]]
|}

== Overview ==
Asynchronous browser agent entry point that executes web browsing tasks using MCP-based browser tools with async LLM orchestration.

=== Description ===
The `infer_async_nestbrowse.py` module provides the main entry point for the NestBrowse async browser agent. It implements an agentic loop that orchestrates browser interactions through the Model Context Protocol (MCP), enabling web page navigation, content extraction, and form filling. The system uses async OpenAI clients for efficient parallel execution.

Key features:
- Async `agentic_loop` function for multi-turn browser interactions
- MCP client integration for browser tool execution
- Configurable max iterations and tool selection
- Support for both single queries and batch processing

=== Usage ===
Use this module as the entry point for running NestBrowse browser automation tasks. Import `agentic_loop` or run as main script with a query parameter.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/NestBrowse/infer_async_nestbrowse.py WebAgent/NestBrowse/infer_async_nestbrowse.py]
* '''Lines:''' 1-203

=== Signature ===
<syntaxhighlight lang="python">
async def agentic_loop(
    messages: List[Dict],
    client: AsyncOpenAI,
    model: str,
    tools: List[Dict],
    max_iterations: int = 30,
    mcp_client: Any = None
) -> Dict:
    """
    Execute async browser agent loop.

    Args:
        messages: Conversation history
        client: AsyncOpenAI client
        model: Model identifier
        tools: MCP tool definitions
        max_iterations: Max interaction steps
        mcp_client: MCP client for browser control

    Returns:
        Dict with final response and trajectory
    """
    ...

async def main(query: str) -> str:
    """
    Main entry point for single query execution.

    Args:
        query: User browser task query

    Returns:
        Final answer string
    """
    ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from WebAgent.NestBrowse.infer_async_nestbrowse import agentic_loop, main
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| messages || List[Dict] || Yes || Conversation messages
|-
| client || AsyncOpenAI || Yes || Async OpenAI client
|-
| model || str || Yes || Model name
|-
| tools || List[Dict] || Yes || MCP tool schemas
|-
| max_iterations || int || No || Max steps (default 30)
|-
| mcp_client || MCPClient || No || Browser control client
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| response || Dict || Contains 'answer', 'trajectory', 'tool_calls'
|-
| answer || str || Final extracted answer
|}

== Usage Examples ==

=== Basic Browser Task ===
<syntaxhighlight lang="python">
import asyncio
from WebAgent.NestBrowse.infer_async_nestbrowse import main

# Run single browser task
answer = asyncio.run(main(
    "Go to https://2025.aclweb.org and find the paper deadline"
))
print(answer)
</syntaxhighlight>

=== Custom Agent Loop ===
<syntaxhighlight lang="python">
import asyncio
from openai import AsyncOpenAI
from WebAgent.NestBrowse.infer_async_nestbrowse import agentic_loop
from WebAgent.NestBrowse.toolkit.mcp_client import mcp_client

async def run_browser_task():
    client = AsyncOpenAI(api_key="your-key")

    async with mcp_client() as mcp:
        tools = mcp.get_tools()
        messages = [
            {"role": "system", "content": "You are a browser automation agent."},
            {"role": "user", "content": "Find the ACL 2025 venue address"}
        ]

        result = await agentic_loop(
            messages=messages,
            client=client,
            model="gpt-4o",
            tools=tools,
            max_iterations=20,
            mcp_client=mcp
        )

        return result['answer']

answer = asyncio.run(run_browser_task())
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_Browser_Agent]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_Python_Dependencies]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_API_Keys_Configuration]]
