# File: `libs/langchain_v1/langchain/agents/middleware/model_call_limit.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 256 |
| Classes | `ModelCallLimitState`, `ModelCallLimitExceededError`, `ModelCallLimitMiddleware` |
| Imports | __future__, langchain, langchain_core, langgraph, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tracks and enforces limits on model API calls during agent execution, preventing runaway loops or excessive API usage at both thread-level (persisted across runs) and run-level (single invocation) granularity.

**Mechanism:** ModelCallLimitMiddleware extends state schema with thread_model_call_count (persisted) and run_model_call_count (UntrackedValue, ephemeral) fields. before_model hook checks if counts >= configured limits (thread_limit, run_limit), responding based on exit_behavior: 'error' raises ModelCallLimitExceededError, 'end' returns Command(jump_to='end') with artificial AIMessage explaining limit exceeded. after_model hook increments both counters after successful calls. Uses @hook_config(can_jump_to=['end']) to enable Command-based flow control.

**Significance:** Essential safety guardrail for production agent deployments - prevents infinite loops from bugs or model confusion from consuming unlimited API quota/cost, provides both short-term (run) and long-term (thread) protection, and offers graceful shutdown vs hard error based on application needs.
