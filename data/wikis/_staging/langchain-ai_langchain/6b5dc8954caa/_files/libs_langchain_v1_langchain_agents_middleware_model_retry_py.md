# File: `libs/langchain_v1/langchain/agents/middleware/model_retry.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 300 |
| Classes | `ModelRetryMiddleware` |
| Imports | __future__, asyncio, langchain, langchain_core, time, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Automatically retries failed model API calls with configurable exponential backoff, filtering retryable exceptions and handling exhaustion with either error propagation or graceful continuation.

**Mechanism:** ModelRetryMiddleware wraps handler in retry loop (initial + max_retries attempts): on exception, checks should_retry_exception against retry_on (tuple of exception types or predicate callable), if retryable and retries remain calculates exponential backoff delay via calculate_delay (initial_delay * backoff_factor^retry_number capped at max_delay with optional ±25% jitter) and sleeps, if exhausted or non-retryable calls _handle_failure which either re-raises (on_failure='error') or returns AIMessage with error details (on_failure='continue' or custom formatter). Uses shared _retry module utilities.

**Significance:** Essential reliability middleware for agents calling rate-limited or flaky LLM APIs - transparently handles transient failures (timeouts, rate limits, 5xx errors) without requiring application-level retry logic, with sophisticated backoff to avoid thundering herd and configurable failure modes balancing robustness (continue with error message) vs strictness (propagate exception).
