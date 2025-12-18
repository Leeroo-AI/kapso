# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_model_call_limit.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 226 |
| Functions | `simple_tool`, `test_middleware_unit_functionality`, `test_thread_limit_with_create_agent`, `test_run_limit_with_create_agent`, `test_middleware_initialization_validation`, `test_exception_error_message`, `test_run_limit_resets_between_invocations` |
| Imports | langchain, langchain_core, langgraph, pytest, tests |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests the ModelCallLimitMiddleware that enforces thread and run limits on model invocations.

**Mechanism:** Uses FakeToolCallingModel and InMemorySaver checkpointer to create test agents, then verifies: (1) middleware correctly tracks thread_model_call_count (cumulative across invocations) and run_model_call_count (per invocation), (2) enforces limits by jumping to "end" node or raising ModelCallLimitExceededError based on exit_behavior, (3) validates initialization parameters (at least one limit required, valid exit_behavior), (4) generates clear error messages showing which limit was exceeded, and (5) run_limit properly resets between invocations while thread_limit accumulates. Tests both isolation unit tests and integration with create_agent.

**Significance:** Ensures agent execution can be constrained to prevent runaway loops and excessive API costs, with proper distinction between per-run limits (prevent infinite loops) and per-thread limits (enforce usage quotas across conversation).
