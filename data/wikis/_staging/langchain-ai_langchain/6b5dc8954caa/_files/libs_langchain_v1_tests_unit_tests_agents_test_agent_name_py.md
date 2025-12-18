# File: `libs/langchain_v1/tests/unit_tests/agents/test_agent_name.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 99 |
| Functions | `simple_tool`, `test_agent_name_set_on_ai_message`, `test_agent_name_not_set_when_none`, `test_agent_name_on_multiple_iterations`, `test_agent_name_async`, `test_agent_name_async_multiple_iterations` |
| Imports | __future__, langchain, langchain_core, model, pytest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests that the `name` parameter in `create_agent` correctly sets the `.name` attribute on AIMessage outputs.

**Mechanism:** Uses FakeToolCallingModel to simulate agent behavior and verifies that when a name is provided to create_agent, all AIMessage instances in the conversation have that name set. Tests both sync and async execution paths, as well as single and multi-turn conversations. Tests cover cases where name is provided (should be set on messages) and when it's not provided (should be None).

**Significance:** This test file ensures proper agent identification in multi-agent systems where different agents need to be distinguished by their names in the message history. Critical for scenarios where multiple agents collaborate and their individual responses must be tracked.
