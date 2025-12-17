# File: `libs/langchain_v1/langchain/agents/middleware/context_editing.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 278 |
| Classes | `ContextEdit`, `ClearToolUsesEdit`, `ContextEditingMiddleware` |
| Imports | __future__, collections, copy, dataclasses, langchain, langchain_core, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Manages conversation context size by automatically clearing older tool results when token limits are exceeded, mirroring Anthropic's context editing capabilities in a model-agnostic way.

**Mechanism:** The module implements a protocol-based design with `ContextEdit` defining the interface for editing strategies. The primary implementation, `ClearToolUsesEdit`, monitors total token count (via `count_tokens_approximately()` or model-specific counting) and triggers when exceeding the `trigger` threshold (default 100K tokens). It identifies `ToolMessage` instances (excluding the most recent `keep` messages and any in `exclude_tools`), replaces their content with a placeholder (`"[cleared]"`), and optionally clears tool input parameters from associated `AIMessage` tool calls. The middleware applies edits via `wrap_model_call()` before each model invocation, using `deepcopy()` to preserve original messages. Cleared messages are marked with `response_metadata.context_editing.cleared` to prevent re-processing.

**Significance:** This middleware solves the critical context window management problem for long-running agent conversations. As agents make many tool calls, keeping all tool results in context can quickly exhaust model token limits (e.g., Claude's 200K limit). By intelligently pruning older tool results while preserving recent ones, agents can maintain conversation continuity without hitting context limits. The approach aligns with Anthropic's `clear_tool_uses_20250919` strategy, making it familiar to Claude users while remaining compatible with any LangChain chat model.
