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

**Purpose:** Test middleware interaction with tools in ModelRequest - filtering, modification, and validation

**Mechanism:** Validates how middleware can control tool availability and handle tool-related scenarios:

**Tool Access Patterns:**
- ModelRequest.tools contains BaseTool objects (not tool node instances)
- Middleware can read tools from request
- Tools are properly typed and accessible

**Tool Modification:**
- Filtering tools to subset (e.g., only allow certain tools)
- Adding tools dynamically based on state
- Removing tools based on conditions (e.g., admin-only tools)
- Setting tools to empty list (valid operation)
- Tool modifications via request.override(tools=new_list)

**Tool Validation:**
- Unknown tools (not in agent's tool registry) raise ValueError
- Error message identifies unknown tool names
- Prevents middleware from adding arbitrary tools

**Tool Composition:**
- Multiple middleware can modify tools in sequence
- Tool modifications propagate through middleware chain
- Later middleware see earlier modifications
- Middleware can provide additional tools via .tools attribute
- Tools from middleware.tools automatically available in agent

**Edge Cases:**
- Empty tool list is valid
- ToolNode instances rejected (must pass tool list)
- Tool filtering doesn't break execution

**Significance:** Ensures middleware can safely control tool access for security, feature gating, and conditional tool availability. Tests validate that tool modifications work correctly through the middleware chain and that the framework prevents invalid tool configurations that could cause runtime errors.
