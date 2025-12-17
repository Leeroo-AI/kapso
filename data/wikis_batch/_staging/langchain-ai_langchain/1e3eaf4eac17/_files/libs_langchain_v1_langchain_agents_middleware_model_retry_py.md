# File: `libs/langchain_v1/langchain/agents/middleware/model_retry.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 300 |
| Classes | `ModelRetryMiddleware` |
| Imports | __future__, asyncio, langchain, langchain_core, time, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Automatically retries failed LLM model calls with exponential backoff and configurable exception filtering, improving agent robustness against transient API failures.

**Mechanism:** The middleware wraps model invocations in a retry loop (initial attempt + `max_retries` additional attempts). It leverages shared utilities from `_retry.py`: `should_retry_exception()` determines if an exception matches the `retry_on` criteria (tuple of types or callable predicate), `calculate_delay()` computes exponential backoff using `initial_delay * (backoff_factor ** retry_number)` with optional jitter, and `validate_retry_params()` ensures configuration validity. On retry exhaustion, the `on_failure` parameter controls behavior: `'error'` re-raises the exception, `'continue'` returns an `AIMessage` with error details (via `_format_failure_message()` or custom callable), allowing the agent to proceed with the failure information in context. The implementation uses `time.sleep()` for sync and `asyncio.sleep()` for async delays.

**Significance:** This middleware is critical for production agent reliability, handling common LLM API issues like rate limits (429), transient server errors (503), and network timeouts. Without retries, agents would fail on temporary glitches that would succeed moments later. The exponential backoff with jitter prevents thundering herd problems when many agents simultaneously retry a recovering service. The flexible `retry_on` filtering allows sophisticated policies: retry rate limits but not authentication errors, retry provider-specific error codes, or use custom predicates based on error messages. The `on_failure='continue'` mode enables self-healing agents that can explain failures to users and attempt alternative approaches, rather than crashing outright.
