# File: `examples/online_serving/openai_responses_client_with_mcp_tools.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 184 |
| Functions | `example_no_filter`, `example_wildcard`, `example_specific_tools`, `example_object_format`, `main` |
| Imports | openai, utils |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** MCP (Model Context Protocol) tools integration with Responses API

**Mechanism:** Demonstrates multiple patterns for filtering MCP server tools: no filter (all tools), wildcard "*" (explicit all), specific tool names list, and object format with tool_names field. Shows integration with code_interpreter and web_search_preview MCP servers. Each example connects to different MCP server labels and URLs, demonstrating tool filtering at the sub-tool level.

**Significance:** Critical example for enterprise deployments using MCP protocol. Shows how vLLM integrates with external tool servers following MCP spec. Important for security (tool filtering), flexibility (multiple server support), and interoperability (standard protocol). Demonstrates vLLM's advanced tool calling architecture beyond simple function definitions.
