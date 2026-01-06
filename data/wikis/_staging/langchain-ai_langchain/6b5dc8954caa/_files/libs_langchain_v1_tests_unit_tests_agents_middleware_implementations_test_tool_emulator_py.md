# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_tool_emulator.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 627 |
| Classes | `FakeModel`, `FakeEmulatorModel`, `TestLLMToolEmulatorBasic`, `TestLLMToolEmulatorMultipleTools`, `TestLLMToolEmulatorModelConfiguration`, `TestLLMToolEmulatorAsync` |
| Functions | `get_weather`, `search_web`, `calculator` |
| Imports | collections, itertools, langchain, langchain_core, pydantic, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests the LLMToolEmulator middleware that replaces real tool execution with LLM-generated synthetic responses for testing and prototyping.

**Mechanism:** Validates selective tool emulation by name or BaseTool instance, emulate-all behavior when tools=None, mixed real/emulated tool execution, custom emulator model configuration (strings or instances), tool filtering to determine which tools get emulated vs executed normally, and both sync/async execution paths. Uses FakeModel and FakeEmulatorModel to simulate tool calling and response generation.

**Significance:** Essential test suite for development workflow middleware that enables rapid prototyping by mocking expensive or unavailable tools while still testing agent logic and flow control.
