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

**Purpose:** Combines reasoning models with tool calling capabilities

**Mechanism:** Uses models like QwQ-32B with both --reasoning-parser and --tool-call-parser flags enabled. Demonstrates that reasoning tokens are NOT parsed during tool calling - only the final output is processed for tool extraction. Tests automatic and named function calling modes in both streaming and non-streaming variants. Extracts reasoning separately from tool calls in streaming mode.

**Significance:** Critical example for advanced reasoning + tool use workflows. Shows how reasoning models can "think" before deciding to call tools, providing transparency into the model's decision-making process. Important for trustworthy AI applications where tool calling rationale is required. Demonstrates the separation of reasoning from tool extraction in vLLM's processing pipeline.
