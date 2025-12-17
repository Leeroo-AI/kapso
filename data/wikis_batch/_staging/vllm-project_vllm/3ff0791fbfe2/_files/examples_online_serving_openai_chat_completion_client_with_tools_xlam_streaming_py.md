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

**Purpose:** xLAM streaming function calling

**Mechanism:** Streaming version of the xLAM function calling example. Handles tool call arguments as they stream in chunks, accumulating them by tool call ID. After streaming completes, executes all tools and sends a follow-up request with results. Demonstrates handling multiple concurrent tool calls in streaming mode.

**Significance:** Shows advanced streaming patterns for xLAM models where tool arguments arrive incrementally. Important for building responsive UIs that display function calls as they're generated.
