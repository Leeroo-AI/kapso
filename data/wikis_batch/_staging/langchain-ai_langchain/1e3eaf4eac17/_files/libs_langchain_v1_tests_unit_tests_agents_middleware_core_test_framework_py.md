# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_framework.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 1048 |
| Classes | `CustomState`, `CustomMiddleware`, `TestAgentMiddlewareHooks`, `TestAgentHooksCombined`, `NoopSeven`, `NoopEight`, `NoopSeven`, `NoopEight`, `NoopZero`, `NoopOne`, `NoopTwo`, `ModifyMiddleware`, `WeatherResponse`, `FakeModelWithBothToolCalls`, `CustomState`, `CustomMiddleware`, `CustomMiddleware`, `MyMiddleware`, `AsyncOnlyMiddleware`, `MixedMiddleware`, `AsyncMiddleware`, `AsyncMiddlewareOne`, `AsyncMiddlewareTwo`, `AsyncMiddlewareOne`, `AsyncMiddlewareTwo`, `MostlySyncMiddleware`, `AsyncMiddleware`, `CustomMiddleware`, `CustomMiddleware` |
| Functions | `test_create_agent_invoke`, `test_create_agent_jump`, `test_simple_agent_graph`, `test_agent_graph_with_jump_to_end_as_after_agent`, `test_on_model_call`, `test_tools_to_model_edge_with_structured_and_regular_tool_calls`, `test_public_private_state_for_custom_middleware`, `test_runtime_injected_into_middleware`, `... +9 more` |
| Imports | collections, langchain, langchain_core, langgraph, pydantic, pytest, syrupy, sys, tests, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive integration testing of the agent middleware framework with full execution flows

**Mechanism:** Tests the complete middleware lifecycle in real agent scenarios:

**Core Framework Tests:**
- Full agent invocations with middleware hooks (before_model, wrap_model_call, after_model)
- Execution order verification through call tracking
- Tool execution integration with middleware
- Jump_to functionality for early exit and conditional routing
- Graph structure validation with snapshot testing

**State Management:**
- Custom state schemas with typed attributes
- Public/private state attributes (OmitFromInput, OmitFromOutput, PrivateStateAttr)
- State injection into tools via InjectedState
- Runtime injection into middleware hooks
- Ephemeral jump_to field handling

**Async/Sync Interoperability:**
- Pure sync middleware on sync path
- Pure async middleware on async path (abefore_model, awrap_model_call, aafter_model)
- Mixed sync/async middleware with appropriate delegation
- NotImplementedError when async-only middleware invoked on sync path
- Sync middleware automatically available in async via delegation (for node hooks)

**Advanced Scenarios:**
- Structured output with tool strategies
- Combined structured and regular tool calls
- Multiple middleware instances working together
- before_agent/after_agent hooks (executed once per thread)
- Model hook repetition across tool calling loops

**Significance:** The most comprehensive middleware test suite - validates that the entire framework integrates correctly with agents, tools, state management, and graph construction. Essential for ensuring middleware behavior is correct in real-world agent execution scenarios, not just in isolation.
