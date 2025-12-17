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

**Purpose:** Tests ToolRetryMiddleware which automatically retries failed tool executions with configurable backoff. Tests validate initialization parameters (max_retries, tools filter, retry_on exceptions, on_failure behavior, backoff configuration), retry logic (succeeding after retries, exhausting retries), exception filtering (type-based and custom predicates), backoff timing (exponential, constant, jitter, max_delay cap), tool-specific filtering, custom failure formatters, and both sync/async execution. Ensures robust handling of transient tool failures.

**Mechanism:** Uses test tools (working_tool always succeeds, failing_tool always fails, TemporaryFailureTool fails N times then succeeds) to simulate various failure scenarios. Validates retry counts by checking ToolMessage content for "N attempts" strings. Tests backoff timing by measuring elapsed time and comparing against expected delays. Validates exception filtering by creating tools that raise different exception types (ValueError, RuntimeError, CustomError) and verifying only matching exceptions are retried. Tests custom failure handlers by providing on_failure as both strings ('error', 'continue') and callable formatters.

**Significance:** Critical for production reliability when tools interact with flaky external services (APIs, databases, network resources). Automatic retry with backoff prevents transient failures from breaking agent flows. Exception filtering allows selective retry (retry network errors but not validation errors). Backoff strategies (exponential with jitter) prevent thundering herd problems. Tool-specific filtering enables different retry policies for different tools. Custom failure formatters support integration with logging/monitoring systems. Proper async support ensures retry delays don't block event loops. Essential for building robust agents that gracefully handle real-world service instability.
