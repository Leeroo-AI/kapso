# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_model_call_limit.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 226 |
| Functions | `simple_tool`, `test_middleware_unit_functionality`, `test_thread_limit_with_create_agent`, `test_run_limit_with_create_agent`, `test_middleware_initialization_validation`, `test_exception_error_message`, `test_run_limit_resets_between_invocations` |
| Imports | langchain, langchain_core, langgraph, pytest, tests |

## Understanding

**Status:** ✅ Explored

**Purpose:** Unit tests for ModelCallLimitMiddleware, validating enforcement of model call limits at both thread (conversation) and run (single invocation) levels. Tests ensure proper limit tracking, exit behaviors, and error messaging for cost control and safety.

**Mechanism:** Tests operate at two levels:
- **Unit-level**: Direct middleware testing with mocked state containing call counts, verifying `before_model` hook returns correct responses when limits are exceeded
- **Integration-level**: Full agent testing with `create_agent` and `InMemorySaver` checkpointer to validate limit enforcement across actual invocations

Key test scenarios:
- **Thread limits**: Accumulated across multiple invocations in same conversation thread
- **Run limits**: Apply per invocation and reset between invocations
- **Exit behaviors**: Tests both "end" (graceful termination with AI message) and "error" (raises ModelCallLimitExceededError)
- **Initialization validation**: Ensures at least one limit specified and valid exit behavior
- **Error messages**: Validates clear messaging indicating which limit was exceeded

Uses `FakeToolCallingModel` to control model behavior and trigger limit conditions predictably.

**Significance:** Essential for production agent deployments to prevent runaway costs and enforce safety guardrails. Thread limits protect against infinite loops across conversations, while run limits prevent single invocations from making excessive model calls. The middleware enables configurable enforcement strategies (graceful vs. error) suitable for different deployment scenarios.
