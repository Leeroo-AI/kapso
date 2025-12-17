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

**Purpose:** Streaming reasoning model responses

**Mechanism:** Streaming variant of reasoning model chat completions. The reasoning tokens stream first, followed by content tokens. Code safely extracts both fields from stream chunks and displays them in real-time. Handles cases where fields may be missing.

**Significance:** Shows how to build UIs that display the model's thinking process as it happens, providing transparency into reasoning models' decision-making in real-time.
