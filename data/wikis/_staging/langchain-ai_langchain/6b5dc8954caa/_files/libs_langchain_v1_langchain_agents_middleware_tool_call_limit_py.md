# File: `libs/langchain_v1/langchain/agents/middleware/tool_call_limit.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 488 |
| Classes | `ToolCallLimitState`, `ToolCallLimitExceededError`, `ToolCallLimitMiddleware` |
| Imports | __future__, langchain, langchain_core, langgraph, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tracks and enforces limits on tool call counts during agent execution with configurable exit behaviors.

**Mechanism:** Implements ToolCallLimitMiddleware that hooks into after_model. Maintains two count dictionaries in state: thread_tool_call_count (persistent) and run_tool_call_count (UntrackedValue, per-run). Separates tool calls into allowed vs blocked based on limits. Supports three exit behaviors: "continue" (injects error ToolMessages for blocked calls), "error" (raises ToolCallLimitExceededError), "end" (jumps to end with error message - requires single tool call). Uses special count_key "__all__" for global limits or tool name for per-tool limits.

**Significance:** Critical safety middleware to prevent runaway agent loops and control resource consumption. Provides flexible limit enforcement at both thread (persistent across runs) and run (per-invocation) levels. Essential for production deployments to cap maximum tool usage and prevent infinite loops or excessive API calls.
