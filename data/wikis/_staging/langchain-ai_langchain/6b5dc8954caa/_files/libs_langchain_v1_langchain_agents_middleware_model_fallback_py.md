# File: `libs/langchain_v1/langchain/agents/middleware/model_fallback.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 135 |
| Classes | `ModelFallbackMiddleware` |
| Imports | __future__, langchain, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides automatic failover to alternative models when primary model calls fail, enabling resilient agent execution across multiple LLM providers or model tiers.

**Mechanism:** ModelFallbackMiddleware accepts ordered sequence of fallback models (strings or BaseChatModel instances), initializing them via init_chat_model if needed. wrap_model_call intercepts handler execution in try/except blocks: attempts primary model from request.model, then iterates through self.models on exception, calling handler(request.override(model=fallback_model)) until success or exhaustion. Re-raises last exception if all models fail. Supports both sync and async via wrap_model_call/awrap_model_call.

**Significance:** Critical reliability middleware for production agents - mitigates provider outages, rate limits, and model-specific failures by transparently switching to alternatives (e.g., GPT-4 → GPT-3.5 → Claude), trading cost/quality for availability without requiring application-level retry logic or error handling.
