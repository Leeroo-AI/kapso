# File: `examples/online_serving/openai_chat_completion_client_with_tools.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 195 |
| Functions | `get_current_weather`, `handle_tool_calls_stream`, `handle_tool_calls_arguments`, `main` |
| Imports | json, openai, typing |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tool calling demonstration with streaming support

**Mechanism:** Shows complete tool calling lifecycle: initial request with tools parameter, streaming tool call responses, parsing chunked tool call arguments, executing local functions, and sending results back for final response. Handles both non-streaming and streaming modes. Includes reasoning field support for reasoning-capable models. Requires specific chat templates and tool-call-parser flags (mistral, hermes).

**Significance:** Reference implementation for function calling / tool use patterns. Critical for building agentic applications where models need to call external functions. Demonstrates proper handling of streaming tool calls (challenging due to incremental JSON parsing) and multi-turn tool conversation patterns. Shows integration between vLLM's tool parsing and OpenAI's tool calling API.
