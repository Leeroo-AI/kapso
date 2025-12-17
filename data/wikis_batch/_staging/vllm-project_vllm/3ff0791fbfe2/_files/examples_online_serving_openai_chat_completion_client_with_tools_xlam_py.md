# File: `examples/online_serving/openai_chat_completion_client_with_tools_xlam.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 245 |
| Functions | `get_weather`, `calculate_expression`, `translate_text`, `process_response`, `run_test_case`, `main` |
| Imports | json, openai, time |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** xLAM function calling example

**Mechanism:** Demonstrates function calling specifically for Salesforce xLAM models (xLAM-2 series). Runs multiple test cases showing weather queries, calculations, and translations. Handles tool calls in non-streaming mode, executes functions locally, and sends results back for model synthesis.

**Significance:** Model-specific example for xLAM-2's function calling capabilities. Shows best practices for multi-tool scenarios and follow-up conversations after tool execution.
