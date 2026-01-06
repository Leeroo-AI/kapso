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

**Purpose:** Tests state_schema parameter for extending AgentState with custom fields. Validates that custom state fields persist through agent execution, work with middleware, and are accessible in tools via ToolRuntime.

**Mechanism:** Creates custom AgentState subclasses with additional fields and tests their preservation through agent invocation. Tests cover single/multiple custom fields, tool runtime access, middleware integration, async operations, and default behavior. Uses FakeToolCallingModel for deterministic testing with predefined tool call sequences.

**Significance:** Critical feature test ensuring users can extend agent state without custom middleware. Enables stateful agent patterns like session tracking, user context, and custom metadata while maintaining type safety and field persistence through the agent execution lifecycle.
