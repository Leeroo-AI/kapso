# File: `examples/online_serving/openai_chat_completion_tool_calls_with_reasoning.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 170 |
| Functions | `get_current_weather`, `extract_reasoning_and_calls`, `main` |
| Imports | openai |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Reasoning models with function calling

**Mechanism:** Demonstrates combining reasoning models (like QwQ-32B) with function calling. The model's internal reasoning process is captured separately from tool calls. Shows both automatic and named function calling modes in streaming and non-streaming variants. The reasoning helps the model decide which tools to use.

**Significance:** Example of advanced agentic patterns where reasoning models can "think through" which functions to call. Useful for complex multi-step tool use scenarios.
