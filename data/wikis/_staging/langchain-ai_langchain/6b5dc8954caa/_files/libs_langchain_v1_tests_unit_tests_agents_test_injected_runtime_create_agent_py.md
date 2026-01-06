# File: `libs/langchain_v1/tests/unit_tests/agents/test_injected_runtime_create_agent.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 831 |
| Classes | `CustomState`, `CustomMiddleware`, `TestMiddleware`, `CustomState`, `CustomState` |
| Functions | `test_tool_runtime_basic_injection`, `test_tool_runtime_async_injection`, `test_tool_runtime_state_access`, `test_tool_runtime_with_store`, `test_tool_runtime_with_multiple_tools`, `test_tool_runtime_config_access`, `test_tool_runtime_with_custom_state`, `test_tool_runtime_no_runtime_parameter`, `... +7 more` |
| Imports | __future__, langchain, langchain_core, langgraph, model, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests ToolRuntime injection functionality in create_agent, verifying tools can access runtime context (state, tool_call_id, config, context, store, stream_writer).

**Mechanism:** Comprehensive test suite covering ToolRuntime injection in various scenarios: basic injection, async execution, state access, store access, multiple tools, config access, custom state schemas, parallel execution, error handling, middleware integration, type hints, name-based injection, and combined injection with InjectedState and InjectedStore. Each test creates tools with a `runtime: ToolRuntime` parameter and verifies the runtime object is properly populated with execution context. Tests verify both sync and async paths, and that tools without runtime parameters continue to work normally.

**Significance:** Essential for advanced tool functionality where tools need access to execution context beyond their explicit parameters. Enables tools to read current state, access persistent storage, understand their call context, and integrate with LangGraph's runtime features. Critical for building context-aware tools that can make decisions based on conversation state or persist data across invocations.
