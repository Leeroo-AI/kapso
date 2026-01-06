# File: `libs/langchain_v1/langchain/agents/middleware/tool_emulator.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 209 |
| Classes | `LLMToolEmulator` |
| Imports | __future__, langchain, langchain_core, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Emulates tool execution using an LLM instead of running actual tools, useful for testing.

**Mechanism:** Implements LLMToolEmulator middleware that intercepts wrap_tool_call. Maintains set of tools to emulate (or emulates all if tools=None). When matched tool is called, builds prompt with tool name/description/args and asks emulator LLM to "generate a realistic response that this tool would return". Uses Claude Sonnet 4.5 by default with temperature=1 for variation. Returns ToolMessage with emulated content, short-circuiting actual tool execution.

**Significance:** Testing/development middleware that enables agent testing without external dependencies or side effects. Allows rapid iteration on agent behavior and prompts without running expensive/slow real tools. Particularly useful for testing tool selection logic, error handling paths, and agent reasoning without requiring full tool infrastructure.
