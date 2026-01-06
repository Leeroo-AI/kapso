# File: `libs/langchain_v1/langchain/agents/middleware/context_editing.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 278 |
| Classes | `ContextEdit`, `ClearToolUsesEdit`, `ContextEditingMiddleware` |
| Imports | __future__, collections, copy, dataclasses, langchain, langchain_core, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Automatically manages conversation context size by clearing older tool results when token counts exceed configurable thresholds, mirroring Anthropic's context editing capabilities in a model-agnostic way.

**Mechanism:** ContextEditingMiddleware applies ContextEdit strategies (currently ClearToolUsesEdit) in wrap_model_call before each model invocation. ClearToolUsesEdit triggers at 100k tokens by default, identifies ToolMessage candidates excluding the most recent N messages and excluded tool names, replaces content with placeholder '[cleared]', optionally clears tool_call args from paired AIMessages, and tracks edits in response_metadata['context_editing']. Token counting uses either count_tokens_approximately (fast O(n) heuristic) or model.get_num_tokens_from_messages (slower, provider-specific).

**Significance:** Essential memory management middleware that prevents context window exhaustion during long-running agent sessions - enables extended conversations by pruning stale tool outputs while preserving recent context, implementing the same strategy as Anthropic's clear_tool_uses_20250919 but compatible with any LangChain chat model.
