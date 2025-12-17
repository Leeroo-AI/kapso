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

**Purpose:** Tests LLMToolEmulator middleware which replaces actual tool execution with LLM-generated mock responses. Tests validate tool filtering (by name or BaseTool instance), selective emulation (some tools emulated, others execute normally), empty/None tools lists (emulate nothing vs emulate all), custom emulator models, and both synchronous and asynchronous execution. Enables testing agents without side effects or external dependencies.

**Mechanism:** Uses FakeModel for the main agent (generates tool calls) and FakeEmulatorModel for generating mock tool responses. Tests specify tools to emulate via tools parameter (list of strings/BaseTool instances, empty list, or None). Validates by checking that emulated tools complete without raising NotImplementedError (which real get_weather/search_web would raise), while non-emulated tools execute normally (calculator returns actual results). Tests verify model configuration by checking custom emulator models are used and default models are initialized when model=None.

**Significance:** Essential for agent testing and development. Tool emulation allows running agents against real models without triggering side effects (sending emails, making API calls, executing commands). Enables rapid iteration by avoiding slow external services and preventing test pollution from actual tool execution. Selective emulation supports testing tool interaction patterns while still executing critical tools. Particularly valuable for integration tests where agent reasoning is tested but tool implementations are not the focus.
