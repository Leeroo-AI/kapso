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

**Purpose:** Function calling with tool support

**Mechanism:** Demonstrates OpenAI-style function calling (tool use) with vLLM. Shows both streaming and non-streaming tool call handling. Models call a weather function, then the client executes it locally and sends results back to continue the conversation. Includes logic for parsing streamed tool call arguments incrementally.

**Significance:** Key example for implementing agentic AI patterns with vLLM. Shows proper tool call orchestration including argument streaming, function execution, and multi-turn tool use conversations.
