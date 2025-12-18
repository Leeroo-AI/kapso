# File: `libs/langchain_v1/langchain/agents/middleware/todo.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 224 |
| Classes | `Todo`, `PlanningState`, `TodoListMiddleware` |
| Functions | `write_todos` |
| Imports | __future__, langchain, langchain_core, langgraph, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides todo list management capabilities for agents to track progress on complex multi-step tasks.

**Mechanism:** Implements TodoListMiddleware that registers a write_todos tool and extends agent state with PlanningState (adds todos field). Uses wrap_model_call to inject WRITE_TODOS_SYSTEM_PROMPT into system messages, guiding agent on when/how to use todos. The write_todos tool accepts list of Todo objects (content, status: pending/in_progress/completed) and returns Command updating state with new todo list. Tool description includes detailed guidance on usage patterns, task states, and when NOT to use (simple/trivial tasks).

**Significance:** Important UX middleware that enables agents to demonstrate thoroughness and provide progress visibility for complex tasks. Helps agents organize multi-step workflows and communicate their planning/execution process to users. Particularly valuable for code generation, refactoring, and other tasks requiring careful planning and sequential execution.
