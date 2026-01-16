# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/qwenvl_dashscope.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 144 |
| Classes | `QwenVLChatAtDS` |
| Imports | copy, dashscope, http, os, pprint, qwen_agent, re, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides a vision-language multimodal LLM client for Qwen-VL models via DashScope's MultiModalConversation API.

**Mechanism:** The `QwenVLChatAtDS` class extends `BaseFnCallModel` with `support_multimodal_input=True`. Key components include: (1) `_chat_stream` and `_chat_no_stream` methods using `dashscope.MultiModalConversation.call()` for streaming/non-streaming multimodal conversations, (2) `_format_local_files` helper that converts local file paths to the required `file://` URI format for images, audio, and video, (3) `_conv_fname` which handles path normalization including Windows path conversion and home directory expansion, (4) `_extract_vl_response` which parses model responses extracting text and box content items, and (5) `_continue_assistant_response` for partial response continuation with the 'partial' flag. Default model is 'qwen-vl-max'.

**Significance:** Core multimodal integration component that serves as the base class for audio (`QwenAudioChatAtDS`) and omni (`QwenOmniChatAtDS`) implementations. Enables the agent framework to process images, videos, and audio in conversations, making it essential for visual search and multimodal reasoning tasks in WebWatcher.
