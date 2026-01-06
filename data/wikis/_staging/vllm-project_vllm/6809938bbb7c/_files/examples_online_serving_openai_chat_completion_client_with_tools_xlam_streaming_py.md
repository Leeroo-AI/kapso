# File: `examples/online_serving/openai_chat_completion_client_with_tools_xlam_streaming.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 273 |
| Functions | `get_weather`, `calculate_expression`, `translate_text`, `process_stream`, `run_test_case`, `main` |
| Imports | json, openai, time |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Streaming version of xLAM tool calling examples

**Mechanism:** Same test suite as xlam.py but with streaming enabled. Handles incremental tool call chunks by tracking tool call IDs and accumulating function arguments across multiple delta updates. Reconstructs complete tool calls from stream, executes functions, and streams follow-up responses. More complex than non-streaming due to stateful argument accumulation.

**Significance:** Shows advanced streaming patterns for function calling. Critical for real-time applications needing tool execution with xLAM models. Demonstrates proper state management for streaming tool calls where function name, arguments, and IDs arrive in separate chunks. Important reference for building responsive agent UIs.
