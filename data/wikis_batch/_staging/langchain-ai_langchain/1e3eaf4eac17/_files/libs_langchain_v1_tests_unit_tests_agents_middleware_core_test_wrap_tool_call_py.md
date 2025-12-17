# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_wrap_tool_call.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 808 |
| Functions | `search`, `calculator`, `failing_tool`, `test_wrap_tool_call_basic_passthrough`, `test_wrap_tool_call_logging`, `test_wrap_tool_call_modify_args`, `test_wrap_tool_call_access_state`, `test_wrap_tool_call_access_runtime`, `... +14 more` |
| Imports | collections, langchain, langchain_core, langgraph, tests |

## Understanding

**Status:** ✅ Explored

**Purpose:** Test wrap_tool_call decorator functionality for intercepting and modifying tool execution

**Mechanism:** Validates the @wrap_tool_call decorator that wraps individual tool calls with a handler pattern:

**Basic Operations:**
- Passthrough without modification
- Logging before/after tool execution
- Call counting per tool

**Tool Call Modification:**
- Modifying tool arguments before execution
- Accessing tool metadata (name, args)
- Mutating tool_call dict in place

**Context Access:**
- Reading agent state from request.state
- Accessing runtime from request.runtime
- Tool call ID and name from request.tool_call

**Error Handling:**
- Retry logic for failing tools (up to N attempts)
- Converting tool errors to error ToolMessages
- Graceful degradation with fallback responses

**Short-Circuiting:**
- Returning custom ToolMessage without calling handler
- Bypassing actual tool execution
- Custom result injection

**Response Modification:**
- Wrapping tool responses with prefixes/suffixes
- Transforming ToolMessage content
- Preserving tool_call_id and name

**Middleware Composition:**
- Multiple wrap_tool_call middleware in sequence
- Outer-inner execution order (outer before, inner before, inner after, outer after)
- Three-level composition nesting
- Outer middleware intercepting inner responses
- Inner middleware short-circuiting (outer still wraps result)
- Mixed passthrough and intercepting handlers

**Decorator Features:**
- Custom middleware names via name parameter
- Tools registration via tools parameter
- Function names as default class names
- Returns ToolMessage or Command objects

**Common Patterns:**
- Caching tool results (key based on name+args)
- Monitoring tool execution time
- Selective tool execution based on conditions
- Multi-level wrapping with interception

**Multiple Tool Calls:**
- Middleware applied to each tool call independently
- Tracking calls across different tools
- Parallel tool execution with middleware

**Significance:** Essential for tool-level observability, caching, retry logic, and security - unlike wrap_model_call which wraps the entire model invocation, wrap_tool_call provides fine-grained control over individual tool executions. Tests ensure the handler pattern works correctly, middleware composes properly, and common use cases (caching, monitoring, error handling) are well-supported.
