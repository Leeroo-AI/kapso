# File: `libs/langchain_v1/langchain/agents/middleware/tool_call_limit.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 488 |
| Classes | `ToolCallLimitState`, `ToolCallLimitExceededError`, `ToolCallLimitMiddleware` |
| Imports | __future__, langchain, langchain_core, langgraph, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Enforces limits on tool call counts at both thread level (persistent across runs) and run level (per invocation), with configurable exit behaviors (continue with error messages, raise exception, or stop immediately) to prevent infinite loops, control costs, and manage agent reliability.

**Mechanism:** The ToolCallLimitMiddleware tracks tool call counts in agent state using a dictionary keyed by tool name (or "__all__" for global limits). In the `after_model` hook, it inspects AIMessage tool calls, separates them into allowed and blocked based on current counts vs limits, increments counters for allowed calls, and handles exceeded limits per the configured exit behavior. The "continue" behavior injects error ToolMessages for blocked calls while letting others proceed; "error" raises ToolCallLimitExceededError; "end" jumps to the end node with a ToolMessage and explanatory AIMessage (but only if no other parallel tool calls are pending). Thread counts persist across agent runs while run counts use UntrackedValue to reset per invocation.

**Significance:** This middleware is critical for production agent systems where unbounded tool execution could cause runaway costs (API calls), infinite loops (recursive tool calling), or resource exhaustion. The dual-level tracking (thread and run) provides fine-grained control: thread limits prevent abuse over time while run limits control per-invocation costs. The per-tool filtering capability allows different limits for different tools (e.g., 5 searches but 20 calculations). The three exit behaviors serve different use cases: continue for graceful degradation, error for strict enforcement, and end for immediate termination with user feedback.
