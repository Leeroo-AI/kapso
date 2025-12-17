# File: `libs/langchain_v1/tests/integration_tests/agents/middleware/test_shell_tool_integration.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 147 |
| Functions | `test_shell_tool_basic_execution`, `test_shell_session_persistence`, `test_shell_tool_error_handling`, `test_shell_tool_with_custom_tools` |
| Imports | __future__, langchain, langchain_core, langgraph, pathlib, pytest, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Integration tests for ShellToolMiddleware with create_agent, covering shell command execution through AI agents.

**Mechanism:** Tests use real AI models (Anthropic Claude, OpenAI GPT-4) to execute shell commands via ShellToolMiddleware. Tests cover basic command execution, session persistence across multiple calls, error handling for invalid commands, and coexistence with custom tools. Uses pytest parametrization for multi-provider testing.

**Significance:** Critical for validating that agents can safely and correctly execute shell commands in isolated workspaces. Tests real-world integration between LLM decision-making and system command execution, ensuring environment variable persistence and proper error reporting.
