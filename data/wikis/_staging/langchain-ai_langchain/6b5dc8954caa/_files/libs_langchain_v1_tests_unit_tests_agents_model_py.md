# File: `libs/langchain_v1/tests/unit_tests/agents/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 109 |
| Classes | `FakeToolCallingModel` |
| Imports | collections, dataclasses, json, langchain_core, pydantic, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides a fake chat model for testing tool calling and structured output functionality.

**Mechanism:** `FakeToolCallingModel` extends `BaseChatModel` with configurable tool call sequences and structured responses. It cycles through predefined tool calls on each invocation, supports both OpenAI and Anthropic tool styles, and can return structured JSON responses. The `bind_tools()` method creates simplified tool specifications for testing without requiring real LLM integration.

**Significance:** Essential testing infrastructure that enables deterministic, fast unit tests of agent behavior without network calls or LLM costs, allowing comprehensive testing of tool calling logic, structured outputs, and conversation flows.
