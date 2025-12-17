# File: `examples/offline_inference/chat_with_tools.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 147 |
| Functions | `generate_random_id`, `get_current_weather` |
| Imports | json, random, string, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates function calling with Mistral models.

**Mechanism:** Initializes LLM with Mistral model using mistral tokenizer/config/load formats. Defines tools schema with get_current_weather function, sends chat messages with tools parameter, parses JSON tool calls from model output, executes corresponding functions, appends tool responses with role="tool", and generates final natural language answer.

**Significance:** Example demonstrating vLLM's function calling capabilities with proper tool schema definition and multi-turn conversation flow.
