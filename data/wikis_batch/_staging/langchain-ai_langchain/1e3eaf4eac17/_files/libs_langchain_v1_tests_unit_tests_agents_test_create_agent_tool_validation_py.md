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

**Purpose:** Tests that tool validation errors only expose LLM-controllable parameters and exclude system-injected parameters (InjectedState, InjectedStore, ToolRuntime) from error messages. This prevents leaking sensitive internal data to the LLM and keeps error feedback focused on what the LLM can actually fix.

**Mechanism:** Creates tools with both LLM-controlled parameters (e.g., `query`, `limit`) and system-injected parameters (e.g., `state`, `store`, `runtime`). Tests simulate invalid tool calls from the model and verify:
- Error messages mention only LLM-controlled parameter names
- Error messages do NOT contain system-injected parameter names or their values
- Sensitive data in state (API keys, passwords, session tokens) never appears in errors
- Both sync and async execution paths properly filter errors
- Tests run on Python < 3.14 (skipped on 3.14+ due to Pydantic model rebuild issues)

**Significance:** Critical security and UX feature that prevents accidental exposure of sensitive system data to LLMs. By filtering validation errors to show only what the LLM controls, the system provides focused, actionable feedback while maintaining security boundaries between LLM-visible and system-internal data.
