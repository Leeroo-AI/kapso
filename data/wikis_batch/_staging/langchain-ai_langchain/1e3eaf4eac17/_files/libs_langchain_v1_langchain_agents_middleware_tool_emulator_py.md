# File: `libs/langchain_v1/langchain/agents/middleware/tool_emulator.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 209 |
| Classes | `LLMToolEmulator` |
| Imports | __future__, langchain, langchain_core, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Emulates tool execution using an LLM instead of actually running tools, primarily for testing and development scenarios where real tool execution is impractical, unavailable, or undesired (e.g., testing without API keys, simulating expensive operations, rapid prototyping).

**Mechanism:** The LLMToolEmulator intercepts tool calls in the `wrap_tool_call` hook, checks if the tool should be emulated (either all tools if tools=None, or specific tools by name), and if so, constructs a prompt describing the tool and its arguments, invokes an LLM (defaults to Claude Sonnet 4.5) to generate a realistic response, and returns a ToolMessage with the emulated content without ever calling the actual tool. The handler callback is never invoked for emulated tools, effectively short-circuiting execution. Non-emulated tools pass through to normal execution via the handler.

**Significance:** This middleware is a valuable development and testing utility that enables agent workflows to be tested without dependencies on external services, APIs, or expensive operations. It's particularly useful for: rapid prototyping (test agent logic before implementing real tools), integration testing (avoid side effects like API calls or database writes), development without credentials (work offline or without API keys), and cost control (avoid charges for expensive API calls during development). The ability to selectively emulate specific tools (by name or BaseTool instance) allows mixing real and emulated tools in the same agent, enabling incremental integration testing.
