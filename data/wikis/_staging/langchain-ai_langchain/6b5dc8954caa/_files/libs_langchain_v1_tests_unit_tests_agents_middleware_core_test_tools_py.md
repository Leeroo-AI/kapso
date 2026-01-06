# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_tools.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 338 |
| Classes | `RequestCapturingMiddleware`, `ToolFilteringMiddleware`, `BadMiddleware`, `AdminState`, `ConditionalToolMiddleware`, `NoToolsMiddleware`, `FirstMiddleware`, `SecondMiddleware`, `ToolProvidingMiddleware` |
| Functions | `test_model_request_tools_are_base_tools`, `test_middleware_can_modify_tools`, `test_unknown_tool_raises_error`, `test_middleware_can_add_and_remove_tools`, `test_empty_tools_list_is_valid`, `test_tools_preserved_across_multiple_middleware`, `test_middleware_with_additional_tools`, `test_tool_node_not_accepted` |
| Imports | collections, langchain, langchain_core, langgraph, pytest, tests |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests middleware tool management capabilities

**Mechanism:** Verifies that middleware can inspect ModelRequest.tools (BaseTool objects), modify the tools list via request.override(), filter/add/remove tools dynamically based on state (e.g., admin permissions), provide additional tools via middleware.tools attribute, handle empty tool lists, preserve modifications across middleware chain, and validate unknown tools raise clear errors while rejecting ToolNode instances.

**Significance:** Validates middleware's ability to control tool availability at runtime, enabling security policies, conditional tool access, tool augmentation, and dynamic tool management patterns critical for production agent deployments.
