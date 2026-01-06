# File: `libs/langchain_v1/tests/unit_tests/agents/test_create_agent_tool_validation.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 379 |
| Classes | `TestState`, `TestState`, `TestState`, `StateWithSecrets` |
| Functions | `test_tool_invocation_error_excludes_injected_state`, `test_tool_invocation_error_excludes_injected_state_async`, `test_create_agent_error_content_with_multiple_params`, `test_create_agent_error_only_model_controllable_params` |
| Imports | langchain, langchain_core, langgraph, model, pytest, sys, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests that tool validation error messages exclude system-injected parameters and only show LLM-controllable arguments.

**Mechanism:** Creates tools with both LLM-controllable parameters (like query, some_val) and system-injected parameters (InjectedState, InjectedStore, ToolRuntime). Deliberately triggers validation errors by providing incorrect arguments, then verifies that error messages only mention the LLM-controllable parameters and never expose system-injected state values or parameter names. Tests cover sync, async, and multiple parameter combinations. Skips on Python 3.14 due to Pydantic model rebuild issues.

**Significance:** Critical security and usability feature that prevents sensitive system data from leaking into error messages shown to LLMs. Ensures LLMs receive focused, actionable feedback they can actually fix without being confused by parameters they don't control. Essential for tools that access sensitive state like API keys, user IDs, or internal system data.
