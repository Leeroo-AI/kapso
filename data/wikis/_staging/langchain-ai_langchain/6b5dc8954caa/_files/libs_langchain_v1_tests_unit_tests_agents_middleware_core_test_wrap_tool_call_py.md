# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_wrap_tool_call.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 808 |
| Functions | `search`, `calculator`, `failing_tool`, `test_wrap_tool_call_basic_passthrough`, `test_wrap_tool_call_logging`, `test_wrap_tool_call_modify_args`, `test_wrap_tool_call_access_state`, `test_wrap_tool_call_access_runtime`, `... +14 more` |
| Imports | collections, langchain, langchain_core, langgraph, tests |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests wrap_tool_call decorator and tool execution interception

**Mechanism:** Validates decorator-based tool call wrapping through 19 test functions covering passthrough, logging, argument modification, state/runtime access, retry on error (with ToolMessage error status), short-circuiting (bypassing tool execution), response modification, multiple middleware composition (2-3 levels with interception patterns), multiple parallel tool calls, custom naming, tools parameter, and practical patterns (caching, monitoring/metrics collection).

**Significance:** Comprehensive validation of the wrap_tool_call hook which enables middleware to intercept, modify, retry, or replace tool executions, critical for implementing tool call monitoring, error recovery, caching, and security policies in production agents.
