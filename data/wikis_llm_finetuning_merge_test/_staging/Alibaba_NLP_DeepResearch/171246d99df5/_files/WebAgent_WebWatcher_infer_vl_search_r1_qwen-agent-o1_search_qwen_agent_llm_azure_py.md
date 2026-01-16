# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/azure.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 41 |
| Classes | `TextChatAtAzure` |
| Imports | openai, os, qwen_agent, typing |

## Understanding

**Status:** Explored

**Purpose:** Provides Azure OpenAI integration for the Qwen agent framework. Enables using Azure-hosted OpenAI models as the LLM backend.

**Mechanism:** The `TextChatAtAzure` class extends `TextChatAtOAI` and overrides the initialization to use Azure-specific authentication. It extracts Azure endpoint from multiple possible config keys (`api_base`, `base_url`, `model_server`, `azure_endpoint`), retrieves the API key from config or the `OPENAI_API_KEY` environment variable, and sets the API version (defaulting to `2024-06-01`). The class creates a custom `_chat_complete_create` method that instantiates an `openai.AzureOpenAI` client with the Azure-specific parameters and delegates chat completion calls to it.

**Significance:** Utility adapter that extends the base OpenAI implementation to support Azure's OpenAI Service. This allows organizations using Azure's enterprise-grade OpenAI deployment to seamlessly integrate with the Qwen agent framework without code changes elsewhere in the system.
