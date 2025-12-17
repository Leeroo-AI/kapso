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

**Purpose:** Provides a todo list management system for agents to track progress on complex multi-step tasks, enabling agents to create structured task lists with status tracking (pending, in_progress, completed) and giving users visibility into agent progress.

**Mechanism:** The TodoListMiddleware registers a `write_todos` tool that agents can call to update their task list, which is stored in the agent state. The tool accepts a list of Todo objects (each with content and status) and updates state using a Command with the new list. The middleware injects a comprehensive system prompt via `wrap_model_call` that guides agents on when and how to use the tool (use for 3+ step tasks, mark in_progress before starting, complete immediately after finishing). The state extension (PlanningState) adds a `todos` field that's omitted from input schema but included in output, making it internal to the agent but visible to users.

**Significance:** This middleware bridges the gap between agent internal planning and user understanding of progress. It encourages agents to break down complex tasks systematically, provides a structured way to track execution progress, and helps users understand what the agent is doing without reading through raw message logs. The detailed prompting about when to use todos (and when not to) prevents overuse for simple tasks while ensuring complex multi-step operations are well-tracked. The customizable system prompts and tool descriptions allow organizations to tailor the planning behavior to their specific needs.
