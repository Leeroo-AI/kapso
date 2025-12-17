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

**Purpose:** MCP tools integration examples

**Mechanism:** Demonstrates Model Context Protocol (MCP) tool integration with various filtering options. Shows four patterns: no filter (all tools), wildcard "*" (explicit all), specific tool names (filtered subset), and object format with additional control. Uses code interpreter and browser MCP tools.

**Significance:** Important example for integrating external tool servers via MCP. Shows how to control which sub-tools within an MCP server are exposed to the model.
