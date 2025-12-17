# File: `libs/langchain_v1/langchain/agents/middleware/_retry.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 123 |
| Functions | `validate_retry_params`, `should_retry_exception`, `calculate_delay` |
| Imports | __future__, collections, random, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides shared retry logic utilities used by both `ModelRetryMiddleware` and `ToolRetryMiddleware`, implementing exponential backoff with jitter and flexible exception filtering.

**Mechanism:** The module defines two key type aliases: `RetryOn` (tuple of exception types or callable predicate) and `OnFailure` (literal actions `'error'/'continue'` or custom formatter callable). Three core functions implement retry behavior: `validate_retry_params()` ensures all numeric parameters (max_retries, delays, backoff_factor) are non-negative; `should_retry_exception()` checks if an exception matches retry criteria using either `isinstance()` checks or custom predicates; `calculate_delay()` computes exponential backoff delays using the formula `initial_delay * (backoff_factor ** retry_number)`, capped at `max_delay`, with optional ±25% random jitter to prevent thundering herd problems.

**Significance:** This module extracts common retry patterns from model and tool execution paths, ensuring consistent behavior across the agent middleware stack. Exponential backoff with jitter is a production-grade reliability pattern that prevents cascading failures and rate limit violations when calling external APIs. The flexible exception filtering allows fine-grained control (e.g., retry 429 rate limits but not 401 auth errors), while the `OnFailure` type enables both graceful degradation (continue with error messages) and fail-fast behavior (re-raise exceptions).
