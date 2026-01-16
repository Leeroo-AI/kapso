# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/oai.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 168 |
| Classes | `TextChatAtOAI` |
| Imports | copy, logging, openai, os, pprint, qwen_agent, typing |

## Understanding

**Status:** Explored

**Purpose:** Implements the OpenAI API-compatible LLM backend through the `TextChatAtOAI` class, supporting both OpenAI's official API and any OpenAI-compatible model server.

**Mechanism:** The class extends `BaseFnCallModel` and initializes with configurable `api_base`/`base_url`/`model_server` and `api_key` (from config or `OPENAI_API_KEY` env var). It supports both OpenAI SDK v0.x and v1.x, handling API differences by moving non-standard params (`top_k`, `repetition_penalty`) to `extra_body`. Key methods include: `_chat_stream` for streaming responses, `_chat_no_stream` for single responses, `_continue_assistant_response` which uses the text completion API for Qwen models (allowing response continuation) or falls back to chat simulation. The `convert_messages_to_dicts` static method transforms Message objects to API-compatible dicts with debug logging.

**Significance:** Primary integration point for OpenAI-compatible APIs. This is the most commonly used LLM backend, supporting OpenAI, local model servers (vLLM, text-generation-inference), and other OpenAI-compatible services. It serves as the base class for the Azure implementation and is central to the framework's flexibility in model deployment.
