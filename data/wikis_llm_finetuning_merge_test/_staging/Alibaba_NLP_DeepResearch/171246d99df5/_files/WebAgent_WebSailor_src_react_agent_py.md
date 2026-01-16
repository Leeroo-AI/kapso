# File: `WebAgent/WebSailor/src/react_agent.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 162 |
| Classes | `MultiTurnReactAgent` |
| Imports | json, openai, os, qwen_agent, tiktoken, transformers, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements the core ReAct (Reasoning + Acting) agent that iteratively thinks about questions, calls search/visit tools, and generates answers for web information seeking tasks.

**Mechanism:** The `MultiTurnReactAgent` class extends Qwen Agent's `FnCallAgent` and implements a multi-turn conversation loop. The `_run()` method orchestrates the agent: it constructs messages with system and user prompts, then loops (up to MAX_LLM_CALL_PER_RUN=40 iterations) calling the LLM via `call_server()` which connects to a local vLLM server. It parses responses for `<tool_call>` tags to extract tool names and arguments, executes tools via `_call_tool()`, and wraps results in `<tool_response>` tags. The loop terminates when an `<answer>` tag is found or limits are reached. Token counting uses AutoTokenizer or tiktoken to enforce MAX_TOKEN_LENGTH (31K) context limits.

**Significance:** The central orchestration component of WebSailor that implements the ReAct paradigm for agentic web search. It bridges the LLM reasoning with tool execution (search, visit) to enable multi-step information gathering.
