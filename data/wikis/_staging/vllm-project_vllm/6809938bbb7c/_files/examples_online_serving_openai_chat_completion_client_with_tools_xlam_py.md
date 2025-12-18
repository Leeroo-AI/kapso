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

**Purpose:** xLAM-2 model tool calling examples with batch processing

**Mechanism:** Demonstrates tool calling with Salesforce xLAM-2 models (xLAM parser). Runs multiple test cases (weather, calculator, translator) with automatic tool execution and follow-up response generation. Collects all tool calls before making a single follow-up request with all results, enabling efficient batch processing. Shows proper message formatting for tool calls and tool results.

**Significance:** Reference for xLAM-2 series function calling models. Demonstrates xLAM-specific tool call parser requirements and efficient multi-tool handling patterns. Important for users choosing xLAM models for function calling tasks, showing best practices for tool result aggregation and multi-turn conversations.
