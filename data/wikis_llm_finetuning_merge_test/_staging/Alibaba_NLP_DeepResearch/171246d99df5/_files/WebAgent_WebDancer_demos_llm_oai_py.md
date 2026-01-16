# File: `WebAgent/WebDancer/demos/llm/oai.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 218 |
| Classes | `TextChatAtOAI` |
| Imports | copy, http, json, logging, openai, os, pprint, qwen_agent, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements `TextChatAtOAI`, an OpenAI-compatible LLM backend for WebDancer agents.

**Mechanism:** Extends `BaseFnCallModel` from qwen_agent and registers as 'oai' LLM type. Key features: (1) `__init__()` - configures OpenAI client with API key (from config or OPENAI_API_KEY env var) and base URL, supports both OpenAI v0.x and v1.x APIs, handles extra parameters (top_k, repetition_penalty) via extra_body, (2) `_chat_stream()` - streams chat responses with delta or full mode, handles reasoning_content (for thinking tokens), and tool_calls (formatted as `<tool_call>` JSON), (3) `_chat_no_stream()` - non-streaming chat completion, (4) `_chat_with_functions()` - function calling support using simulated response completion. Raises ValueError if API key is missing or 'EMPTY'.

**Significance:** Critical LLM integration component. Enables WebDancer to use any OpenAI-compatible API (including local vLLM servers), making it flexible for different deployment scenarios.
