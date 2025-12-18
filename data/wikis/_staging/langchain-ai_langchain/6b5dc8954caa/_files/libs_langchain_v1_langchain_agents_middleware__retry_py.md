# File: `libs/langchain_v1/langchain/agents/middleware/_retry.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 123 |
| Functions | `validate_retry_params`, `should_retry_exception`, `calculate_delay` |
| Imports | __future__, collections, random, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Shared retry logic utilities used by both ModelRetryMiddleware and ToolRetryMiddleware, providing exception filtering, backoff calculation, and failure handling configuration.

**Mechanism:** Defines type aliases RetryOn (exception tuple or predicate callable) and OnFailure (literal 'error'/'continue' or error message formatter callable). Provides three utility functions: validate_retry_params checks for non-negative values, should_retry_exception tests exceptions against RetryOn criteria using isinstance or callable checks, calculate_delay implements exponential backoff with formula initial_delay * (backoff_factor ** retry_number) capped at max_delay with optional ±25% jitter to prevent thundering herd.

**Significance:** DRY abstraction layer that eliminates code duplication between model and tool retry implementations - ensures consistent retry behavior and configuration validation across both middleware types while keeping the retry algorithms model/tool-agnostic.
