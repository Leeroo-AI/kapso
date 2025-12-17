# File: `libs/langchain_v1/langchain/agents/middleware/model_fallback.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 135 |
| Classes | `ModelFallbackMiddleware` |
| Imports | __future__, langchain, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides automatic failover to backup LLM models when the primary model fails, improving agent reliability and availability through sequential fallback chains.

**Mechanism:** The middleware accepts a sequence of fallback models (strings like `"openai:gpt-4o-mini"` or `BaseChatModel` instances) and initializes them using `init_chat_model()`. In `wrap_model_call()`, it catches all exceptions from the primary model (configured in `create_agent()`), then iterates through fallback models using `request.override(model=fallback_model)` to substitute the model while preserving messages, tools, and other request parameters. If all models fail, it re-raises the last exception. The async version `awrap_model_call()` provides identical logic with `await` for async handlers.

**Significance:** This middleware is essential for production systems requiring high availability despite LLM API outages or rate limits. It enables graceful degradation strategies like: (1) expensive-to-cheap fallback (GPT-4 → GPT-3.5 → local model), (2) provider redundancy (OpenAI → Anthropic → Cohere), (3) region failover for geo-distributed deployments. Unlike simple retries, fallback switches models entirely, useful when a specific model/provider has systemic issues. The transparent model substitution via `request.override()` maintains full compatibility with LangChain's tool binding, streaming, and structured output features. Combined with `ModelRetryMiddleware`, it forms a comprehensive error handling strategy: retry for transient errors, fallback for persistent ones.
