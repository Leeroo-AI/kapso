# File: `libs/langchain_v1/tests/unit_tests/agents/test_agent_name.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 99 |
| Functions | `simple_tool`, `test_agent_name_set_on_ai_message`, `test_agent_name_not_set_when_none`, `test_agent_name_on_multiple_iterations`, `test_agent_name_async`, `test_agent_name_async_multiple_iterations` |
| Imports | __future__, langchain, langchain_core, model, pytest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests that the `name` parameter in `create_agent()` correctly sets the `.name` field on all AIMessage outputs produced by the agent. Covers both single-turn and multi-turn conversations in sync and async modes.

**Mechanism:** Uses `FakeToolCallingModel` to create deterministic agent scenarios. Tests verify:
- When `name="test_agent"` is provided, all AIMessages have `.name == "test_agent"`
- When no name is provided, all AIMessages have `.name == None`
- Name is consistently applied across multiple agent iterations (tool calls)
- Behavior is identical in async execution paths

**Significance:** Ensures that agent identity/naming is properly propagated through the message history. This is important for multi-agent systems where different agents need to be distinguished in conversation logs, and for systems that use message names for routing or attribution.
