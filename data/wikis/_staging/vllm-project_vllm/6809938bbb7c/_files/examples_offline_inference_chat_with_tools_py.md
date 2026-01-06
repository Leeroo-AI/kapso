# File: `examples/offline_inference/chat_with_tools.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 147 |
| Functions | `generate_random_id`, `get_current_weather` |
| Imports | json, random, string, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Offline demonstration of function calling with vLLM using Mistral models that support tool use.

**Mechanism:** Defines tool schemas (e.g., get_current_weather function) and uses llm.chat() with tools parameter. Model generates JSON tool calls, which are parsed and executed, with results fed back as tool messages for final response generation. Demonstrates the complete tool-calling loop with Mistral's native format.

**Significance:** Shows how to implement function calling / tool use in offline mode, enabling LLMs to interact with external APIs and functions. Critical pattern for building agents and interactive systems that need to call external tools based on user requests.
