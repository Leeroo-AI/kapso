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

**Purpose:** Comprehensive integration tests for agent middleware framework

**Mechanism:** Tests end-to-end agent execution with middleware through invoke/ainvoke, verifying hook execution order, state management (public/private/omitted attributes), runtime injection, tool execution, jump_to control flow, structured output handling, async/sync execution paths, before_agent/after_agent lifecycle hooks, and proper error handling for incompatible sync/async middleware combinations using both class-based and decorator middleware.

**Significance:** Core test suite validating the complete middleware framework integration with agents, ensuring all middleware features work correctly in real agent execution scenarios including checkpointing, tool calling, state isolation, and graph-based control flow.
