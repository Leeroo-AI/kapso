# File: `libs/langchain_v1/langchain/agents/middleware/model_call_limit.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 256 |
| Classes | `ModelCallLimitState`, `ModelCallLimitExceededError`, `ModelCallLimitMiddleware` |
| Imports | __future__, langchain, langchain_core, langgraph, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tracks and enforces limits on the number of LLM calls agents can make, preventing runaway costs and infinite loops through configurable per-thread and per-run limits.

**Mechanism:** The middleware extends `AgentState` with two counters: `thread_model_call_count` (persisted across agent invocations) and `run_model_call_count` (annotated with `UntrackedValue` to reset per-run). In `before_model()`, it checks if current counts exceed configured `thread_limit` or `run_limit`. When limits are hit, it either raises `ModelCallLimitExceededError` (if `exit_behavior='error'`) or uses LangGraph's Command system to jump to the `'end'` node while injecting an `AIMessage` explaining the limit exceeded (if `exit_behavior='end'`). The `after_model()` hook increments both counters after successful model calls. The `@hook_config(can_jump_to=["end"])` decorator registers the jump capability with LangGraph.

**Significance:** This middleware provides critical cost and safety controls for production agent deployments. Without limits, agents can enter infinite loops (e.g., repeatedly retrying failed tools) or rack up excessive API costs from chatty behaviors. Thread-level limits enable quotas per conversation (e.g., "10 model calls per customer session"), while run-level limits prevent single invocations from burning through budgets. The dual exit behaviors support different use cases: `'end'` for graceful degradation in user-facing apps, `'error'` for strict enforcement in testing/CI pipelines. The private state attributes (`PrivateStateAttr`) ensure counters don't leak into application-visible state.
