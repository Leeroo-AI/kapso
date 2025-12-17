# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_todo.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 520 |
| Functions | `test_todo_middleware_initialization`, `test_has_write_todos_tool`, `test_todo_middleware_default_prompts`, `test_adds_system_prompt_when_none_exists`, `test_appends_to_existing_system_prompt`, `test_todo_middleware_on_model_call`, `test_custom_system_prompt`, `test_todo_middleware_custom_system_prompt`, `... +11 more` |
| Imports | __future__, langchain, langchain_core, langgraph, pytest, tests, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests TodoListMiddleware which provides agents with task planning and tracking capabilities through a write_todos tool. Tests validate tool registration, system prompt injection (adding/appending), custom prompts and descriptions, tool execution with validation, state updates, and both synchronous and asynchronous execution. Ensures agents can manage structured todo lists with pending/in_progress/completed states.

**Mechanism:** Tests use wrap_model_call to capture modified model requests and verify system prompts are correctly injected/appended without mutating original requests. Creates agents with FakeToolCallingModel pre-programmed with tool call sequences (write_todos calls with different todo states). Validates tool execution by invoking write_todos directly with test tool calls and checking state updates. Tests parametrize over different prompt configurations (None, existing, custom) and todo list formats (empty, single task, multiple tasks with various statuses).

**Significance:** Enables structured task management for multi-step agent operations. The middleware automatically registers the write_todos tool and injects guidance prompts, making task tracking transparent to agent developers. Validation ensures todo items have required fields (content, status) and valid status values. The test demonstrates a complete middleware pattern: tool registration, system prompt modification, state schema extension (PlanningState), and tool execution with state updates. Critical for building agents that can break down complex requests into manageable subtasks and track progress.
