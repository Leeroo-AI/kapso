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

**Purpose:** Responses API with function calling

**Mechanism:** Combines the Responses API format with function calling. Model calls a weather function, client executes it and returns results as a function_call_output message type. Uses required tool choice to force function calling.

**Significance:** Example showing how to do function calling using the newer Responses API conventions rather than the chat completions format.
