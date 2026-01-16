# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/qwenvl_oai.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 59 |
| Classes | `QwenVLChatAtOAI` |
| Imports | copy, logging, os, pprint, qwen_agent, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides a vision-language multimodal LLM client for Qwen-VL models using OpenAI-compatible API format instead of DashScope's native format.

**Mechanism:** The `QwenVLChatAtOAI` class extends `TextChatAtOAI` with `support_multimodal_input=True`. The key method `convert_messages_to_dicts` transforms qwen-agent's internal message format to OpenAI's vision API format: (1) Text content items become `{'type': 'text', 'text': v}`, (2) Image content items become `{'type': 'image_url', 'image_url': {'url': v}}`, (3) Local images (not starting with http/https/data) are encoded to base64 using `encode_image_as_base64` with max short side of 1080px, (4) `file://` prefixes are stripped from local paths. Debug logging truncates base64 image URLs to 64 chars for readability. Registered with `@register_llm('qwenvl_oai')`.

**Significance:** Provides an alternative integration path for Qwen-VL models through OpenAI-compatible APIs, enabling deployment flexibility. This allows using Qwen-VL with any OpenAI-compatible server (vLLM, SGLang, etc.) rather than being locked to DashScope, important for self-hosted or alternative cloud deployments.
