# File: `libs/langchain_v1/tests/integration_tests/agents/middleware/test_shell_tool_integration.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 147 |
| Functions | `test_shell_tool_basic_execution`, `test_shell_session_persistence`, `test_shell_tool_error_handling`, `test_shell_tool_with_custom_tools` |
| Imports | __future__, langchain, langchain_core, langgraph, pathlib, pytest, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Integration tests for ShellToolMiddleware functionality with LangGraph agents, verifying shell command execution, session persistence, error handling, and compatibility with custom tools across multiple LLM providers.

**Mechanism:** Uses pytest parametrization to test ShellToolMiddleware with different chat models (Anthropic Claude, OpenAI GPT). Creates agents via `create_agent()` with middleware attached, invokes them with HumanMessages requesting shell operations, and asserts on tool message outputs. Tests cover: basic echo commands, environment variable persistence across calls, error capture from invalid commands, and integration with custom tools. All tests use temporary workspaces.

**Significance:** Critical integration test suite ensuring ShellToolMiddleware works correctly in real-world agent scenarios. Validates cross-provider compatibility and complex interaction patterns like session state management and mixed tool usage, which are essential for production agent deployments requiring shell access.
