# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_decorators.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 757 |
| Classes | `CustomState`, `JumpMiddleware`, `EmptyMiddleware`, `SyncOnlyMiddleware`, `AsyncOnlyMiddleware` |
| Functions | `test_tool`, `test_before_model_decorator`, `test_after_model_decorator`, `test_on_model_call_decorator`, `test_all_decorators_integration`, `test_decorators_use_function_names_as_default`, `test_hook_config_decorator_on_class_method`, `test_can_jump_to_with_before_model_decorator`, `... +23 more` |
| Imports | langchain, langchain_core, langgraph, pytest, syrupy, tests, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive tests for middleware decorator functions

**Mechanism:** Tests five decorator patterns (before_model, after_model, wrap_model_call, wrap_tool_call, dynamic_prompt) in both sync and async modes, verifying decorator configuration options (state_schema, tools, can_jump_to, name), function name preservation, integration with create_agent, and proper handling of jump_to graph control flow including snapshot testing of generated Mermaid diagrams.

**Significance:** Validates the decorator-based middleware API which provides a simpler alternative to class-based middleware, ensuring decorators properly create AgentMiddleware instances, handle both sync/async execution paths, and integrate correctly with the agent framework including graph-based control flow.
