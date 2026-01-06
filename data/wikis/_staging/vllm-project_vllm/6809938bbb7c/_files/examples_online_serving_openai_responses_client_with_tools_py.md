# File: `examples/online_serving/openai_responses_client_with_tools.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 83 |
| Functions | `get_weather`, `main` |
| Imports | json, openai, utils |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Function calling example using Responses API

**Mechanism:** Uses /v1/responses endpoint with tools parameter and tool_choice="required". Shows complete tool calling flow: initial request returns function_call messages, execute function locally, append both function_call and function_call_output messages to input, make follow-up request for final answer. Requires structured outputs backend (xgrammar) and tool-call-parser (hermes).

**Significance:** Demonstrates tool calling through the Responses API rather than Chat Completions API. Important for understanding vLLM's support for OpenAI's newer API conventions. Shows the different message format (function_call type vs tool_calls field) used by Responses API. Useful for applications standardizing on Responses API for reasoning models with tools.
