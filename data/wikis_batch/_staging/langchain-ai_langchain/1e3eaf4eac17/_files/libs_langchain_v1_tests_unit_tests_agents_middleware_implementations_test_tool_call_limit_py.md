# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_tool_call_limit.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 797 |
| Functions | `test_middleware_initialization_validation`, `test_middleware_name_property`, `test_middleware_unit_functionality`, `test_middleware_end_behavior_with_unrelated_parallel_tool_calls`, `test_middleware_with_specific_tool`, `test_middleware_error_behavior`, `test_multiple_middleware_instances`, `test_run_limit_with_multiple_human_messages`, `... +9 more` |
| Imports | langchain, langchain_core, langgraph, pytest, tests |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests ToolCallLimitMiddleware which prevents agents from exceeding tool call budgets across thread (total) and run (per-invocation) scopes. Tests validate initialization with thread_limit/run_limit, exit behaviors (error/end/continue), tool-specific filtering, parallel tool call handling, count tracking (excluding blocked calls from thread count), message injection (error ToolMessages + explanatory AIMessages), and limit reached vs exceeded semantics.

**Mechanism:** Tests create middleware instances with various configurations and invoke agents with FakeToolCallingModel generating predetermined tool call sequences. Validates count tracking by checking state updates (thread_tool_call_count, run_tool_call_count). Tests exit behaviors by verifying error raising, jump_to commands, or continued execution. Validates message injection by checking for error ToolMessages (sent to model) and AIMessages (displayed to user) with appropriate context. Tests parallel calls by proposing multiple simultaneous tool calls and verifying only allowed calls increment thread count while all increment run count.

**Significance:** Prevents runaway agents from exhausting resources or costs through excessive tool calls. Thread limit provides a hard cap across entire conversation, while run limit allows resetting quotas per user message. Tool-specific filtering enables different limits for different tools (e.g., allow unlimited calculator calls but limit expensive search API calls). Exit behaviors provide flexibility: 'error' for hard failures, 'end' for graceful stops, 'continue' for soft blocking. Critical for production deployment where tool calls may have cost, rate limits, or security implications.
