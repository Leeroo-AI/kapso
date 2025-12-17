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

**Purpose:** Comprehensive testing of middleware decorators (@before_model, @after_model, @wrap_model_call, @dynamic_prompt) with sync/async support

**Mechanism:** Tests middleware decorator functionality across multiple dimensions:

**Core Decorator Tests:**
- `@before_model` - executes before model invocation with optional jump_to support
- `@after_model` - executes after model invocation with optional jump_to support
- `@wrap_model_call` - wraps model calls for retry/modification
- `@dynamic_prompt` - dynamically generates system prompts based on request state
- `@hook_config` - adds can_jump_to metadata to middleware methods

**Decorator Features:**
- Custom state schemas with state_schema parameter
- Tool registration via tools parameter
- Custom naming with name parameter
- Function names used as default class names
- Jump destinations configured via can_jump_to parameter

**Sync/Async Support:**
- Pure sync decorators work in sync mode only
- Pure async decorators work in async mode only
- Mixed middleware with both sync/async methods work on both paths
- Async decorators raise NotImplementedError on sync invocation path
- Sync decorators automatically delegate in async mode (for node hooks only)

**Integration Tests:**
- Multiple decorators working together in full agents
- Graph structure snapshots verify correct conditional edges
- State modifications and jump_to behavior validation
- Runtime injection verification

**Significance:** Essential for validating the decorator-based API that makes middleware creation ergonomic - developers can write simple functions with decorators instead of creating full middleware classes. Tests ensure async/sync compatibility and proper metadata preservation for graph construction.
