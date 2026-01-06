# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_todo.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 520 |
| Functions | `test_todo_middleware_initialization`, `test_has_write_todos_tool`, `test_todo_middleware_default_prompts`, `test_adds_system_prompt_when_none_exists`, `test_appends_to_existing_system_prompt`, `test_todo_middleware_on_model_call`, `test_custom_system_prompt`, `test_todo_middleware_custom_system_prompt`, `... +11 more` |
| Imports | __future__, langchain, langchain_core, langgraph, pytest, tests, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests the TodoListMiddleware that enables agents to maintain and update structured task lists through a write_todos tool.

**Mechanism:** Validates middleware initialization with default/custom prompts, system prompt injection/appending behavior, write_todos tool registration and execution, state updates via PlanningState schema, input validation for todo items with required fields (content/status), and both sync/async model call wrapping. Tests agent integration showing todo progression from pending to completed states.

**Significance:** Key test suite for task planning middleware that allows agents to track multi-step workflows, demonstrating how middleware can extend agent capabilities with stateful tools and custom state schemas.
