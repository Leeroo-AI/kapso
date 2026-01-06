# File: `examples/online_serving/openai_chat_completion_with_reasoning_streaming.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 73 |
| Functions | `main` |
| Imports | openai |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Streaming variant of reasoning model chat completion

**Mechanism:** Streams chat completions from reasoning models, printing reasoning and content as they arrive. Carefully checks for existence of reasoning and content attributes in delta objects since they may not always be present. Separates display of reasoning output from content output with clear visual indicators. Uses getattr with fallback to safely extract optional fields.

**Significance:** Enables real-time display of model's thinking process. Critical for interactive applications where users want to see reasoning unfold in real-time rather than waiting for complete response. Shows proper error handling for streaming responses where field availability varies across chunks. Important UX pattern for reasoning-enabled chatbots.
