# File: `examples/online_serving/openai_chat_completion_client_with_tools_required.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 130 |
| Functions | `main` |
| Imports | openai |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Required tool choice demonstration

**Mechanism:** Shows how to force the model to call tools using `tool_choice="required"` parameter. Demonstrates both streaming and non-streaming modes where the model must select and call one or more tools rather than responding directly. Uses weather and forecast functions.

**Significance:** Example of constrained generation with function calling - useful when you need guaranteed structured output or tool usage rather than free-form text responses.
