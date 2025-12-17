# File: `libs/langchain_v1/tests/unit_tests/agents/test_state_schema.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 189 |
| Classes | `CustomState`, `CustomState`, `ExtendedState`, `UserState`, `MiddlewareState`, `TestMiddleware`, `AsyncState` |
| Functions | `simple_tool`, `test_state_schema_single_custom_field`, `test_state_schema_multiple_custom_fields`, `test_state_schema_with_tool_runtime`, `test_state_schema_with_middleware`, `test_state_schema_none_uses_default`, `test_state_schema_async` |
| Imports | __future__, langchain, langchain_core, model, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests the `state_schema` parameter in `create_agent`, validating that custom state fields can extend `AgentState` and are preserved through agent execution, accessible in tools and middleware.

**Mechanism:** Implements seven test scenarios covering different aspects of state schema functionality:
1. `test_state_schema_single_custom_field`: Basic custom field preservation
2. `test_state_schema_multiple_custom_fields`: Multiple custom fields (user_id, session_id, context)
3. `test_state_schema_with_tool_runtime`: Custom fields accessible via ToolRuntime in tool execution
4. `test_state_schema_with_middleware`: State schema merges with middleware state schemas
5. `test_state_schema_none_uses_default`: None value falls back to default AgentState
6. `test_state_schema_async`: Async agent and tool support with custom state

All tests use `FakeToolCallingModel` to control tool execution and verify that custom state fields are preserved in the result dictionary and accessible throughout the agent lifecycle.

**Significance:** Enables developers to extend agent state without creating custom middleware, providing a cleaner API for adding context like user IDs, session data, or application-specific state. Ensures state fields are accessible in tools (via ToolRuntime) and middleware, and that multiple state schemas can be merged when middleware also defines state schemas.
