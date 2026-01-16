# File: `inference/react_agent.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 247 |
| Classes | `MultiTurnReactAgent` |
| Functions | `today_date` |
| Imports | asyncio, datetime, json, json5, openai, os, prompt, qwen_agent, random, time, ... +7 more |

## Understanding

**Status:** Explored

**Purpose:** Implements the core ReAct (Reasoning + Acting) agent loop that powers the DeepResearch system, orchestrating multi-turn conversations with tool execution to answer complex research questions.

**Mechanism:** The `MultiTurnReactAgent` class extends `FnCallAgent` and implements a reasoning loop in `_run()`: 1) Initializes conversation with system prompt (from prompt.py) containing current date; 2) Iteratively calls the LLM server, parses responses for `<tool_call>` tags; 3) Dispatches to appropriate tools (Search, Visit, Scholar, PythonInterpreter, FileParser) via `custom_call_tool()`; 4) Appends tool results wrapped in `<tool_response>` tags; 5) Continues until `<answer>` tags are found, token limit (110K) is reached, or max LLM calls (100) exhausted. Includes exponential backoff retry logic and time limits (150 minutes).

**Significance:** The central brain of the DeepResearch system. Implements the agentic reasoning pattern that enables autonomous research by combining LLM reasoning with tool execution in a feedback loop until the research question is answered.
