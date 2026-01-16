# File: `WebAgent/WebDancer/demos/llm/qwen_dashscope.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 140 |
| Classes | `QwenChatAtDS` |
| Functions | `initialize_dashscope` |
| Imports | dashscope, http, os, pprint, qwen_agent, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements `QwenChatAtDS`, a DashScope-based LLM backend for using Alibaba's Qwen models.

**Mechanism:** Extends `BaseFnCallModel` and registers as 'qwen_dashscope' LLM type. Key features: (1) `__init__()` - initializes with dashscope SDK, defaults to 'qwen-max' model, calls `initialize_dashscope()` for API setup, (2) `_chat_stream()` - uses dashscope.Generation.call with stream=True, supports both delta and full stream output modes, handles reasoning_content for thinking tokens, (3) `_chat_no_stream()` - non-streaming chat completion, (4) `_delta_stream_output()` and `_full_stream_output()` - static methods that process streaming chunks differently (delta yields incremental content, full accumulates). The `initialize_dashscope()` function configures API key from config or DASHSCOPE_API_KEY env var, and optional HTTP/WebSocket base URLs.

**Significance:** Native Alibaba Cloud integration for WebDancer. Provides optimized access to Qwen models via DashScope API, useful for production deployments on Alibaba infrastructure.
