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

**Purpose:** Comprehensive test suite verifying that ToolRuntime injection works correctly in `create_agent()` across various scenarios. ToolRuntime provides tools with access to state, config, store, context, and stream_writer without requiring LLMs to pass these values.

**Mechanism:** Tests cover 15+ scenarios including:
- Basic runtime injection (state, tool_call_id, config, context, store, stream_writer)
- Async tool execution with runtime access
- Multiple tools accessing runtime simultaneously
- Runtime used with custom state schemas and middleware
- Tools without runtime parameters (mixed tool sets)
- Parallel tool execution with runtime injection
- Error handling with runtime access
- Name-based injection (parameter named 'runtime' without type annotation)
- Combined injection (InjectedState + ToolRuntime + InjectedStore working together)
- Explicit args_schema verification (injected params excluded from schema)

**Significance:** Critical for ensuring the dependency injection system works reliably. ToolRuntime allows tools to access execution context and system resources without polluting the tool schema exposed to LLMs. These tests verify that the injection mechanism works correctly across sync/async execution, parallel tool calls, custom state, middleware, and when combined with other injection types.
