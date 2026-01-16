# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/qwen_dashscope.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 144 |
| Classes | `QwenChatAtDS` |
| Functions | `initialize_dashscope` |
| Imports | dashscope, http, os, pprint, qwen_agent, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides a text-based LLM client for Qwen models via Alibaba's DashScope API, enabling chat completions with streaming support.

**Mechanism:** The `QwenChatAtDS` class extends `BaseFnCallModel` and implements chat methods using `dashscope.Generation.call()`. Key features include: (1) `_chat_stream` and `_chat_no_stream` methods for streaming/non-streaming chat completions, (2) `_delta_stream_output` which implements a delay buffer to smooth streaming output, (3) `_full_stream_output` for complete response streaming, (4) `continue_assistant_response` for text completion continuation using raw prompts, and (5) `initialize_dashscope` helper function that configures API keys and URLs from config or environment variables (`DASHSCOPE_API_KEY`, `DASHSCOPE_HTTP_URL`, `DASHSCOPE_WEBSOCKET_URL`). The default model is 'qwen-max'.

**Significance:** Core LLM integration component for text-only Qwen models in the qwen-agent framework. This serves as the foundation for other modality-specific implementations (audio, vision, omni) and provides the DashScope initialization logic reused across the codebase.
