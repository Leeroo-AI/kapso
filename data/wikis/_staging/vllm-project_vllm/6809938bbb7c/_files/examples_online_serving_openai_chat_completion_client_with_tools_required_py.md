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

**Purpose:** Demonstrates tool_choice="required" parameter with structured outputs

**Mechanism:** Configures vLLM with structured outputs backend (outlines or xgrammar) and uses tool_choice="required" to force the model to call a tool rather than generating free text. Shows both streaming and non-streaming modes with multiple tool definitions. The "required" mode ensures deterministic tool calling behavior.

**Significance:** Important for scenarios requiring guaranteed tool invocation (e.g., structured data extraction, API gateways). Shows how to combine vLLM's structured output capabilities with tool calling constraints. Useful for reliable agent workflows where free-form responses are unacceptable.
