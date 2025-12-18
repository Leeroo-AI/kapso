# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_tool_call_limit.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 797 |
| Functions | `test_middleware_initialization_validation`, `test_middleware_name_property`, `test_middleware_unit_functionality`, `test_middleware_end_behavior_with_unrelated_parallel_tool_calls`, `test_middleware_with_specific_tool`, `test_middleware_error_behavior`, `test_multiple_middleware_instances`, `test_run_limit_with_multiple_human_messages`, `... +9 more` |
| Imports | langchain, langchain_core, langgraph, pytest, tests |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests the ToolCallLimitMiddleware that restricts tool execution to prevent runaway agents by enforcing thread and run-scoped call limits.

**Mechanism:** Validates initialization with thread_limit/run_limit constraints, tool-specific vs global limiting, exit behaviors (error/end/continue), parallel tool call handling, count tracking that excludes blocked calls from thread totals but includes them in run totals, artificial error message injection for blocked calls, and run limit reset between invocations. Tests edge cases like parallel mixed tool calls, duplicate tracking, and deprecation warnings.

**Significance:** Critical safety test suite ensuring agents respect configurable execution boundaries, preventing infinite loops and excessive API costs while maintaining granular control over specific tools.
