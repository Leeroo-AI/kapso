# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_tool_retry.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 1007 |
| Classes | `TemporaryFailureTool`, `CustomError` |
| Functions | `working_tool`, `failing_tool`, `test_tool_retry_initialization_defaults`, `test_tool_retry_initialization_custom`, `test_tool_retry_initialization_with_base_tools`, `test_tool_retry_initialization_with_mixed_tools`, `test_tool_retry_invalid_max_retries`, `test_tool_retry_invalid_initial_delay`, `... +25 more` |
| Imports | collections, langchain, langchain_core, langgraph, pytest, tests, time |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests the ToolRetryMiddleware that automatically retries failed tool executions with configurable backoff strategies and exception filtering.

**Mechanism:** Validates initialization parameters (max_retries, backoff_factor, initial_delay, max_delay, jitter), tool-specific vs global retry policies, exception type filtering (tuple or callable), on_failure behaviors (continue/error/custom formatter), exponential/constant backoff timing, retry attempt counting, and both sync/async execution paths. Tests successful recovery after temporary failures, backoff delay calculations with jitter, and composition with other middleware.

**Significance:** Critical resilience test suite ensuring agents can recover from transient tool failures (network issues, rate limits) through intelligent retry mechanisms, preventing workflow failures from temporary errors.
