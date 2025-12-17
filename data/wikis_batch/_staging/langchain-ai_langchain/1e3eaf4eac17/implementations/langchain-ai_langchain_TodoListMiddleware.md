# TodoListMiddleware Implementation

## Metadata
- **Component**: `TodoListMiddleware`
- **Package**: `langchain.agents.middleware.todo`
- **File Path**: `/tmp/praxium_repo_wjjl6pl8/libs/langchain_v1/langchain/agents/middleware/todo.py`
- **Type**: Agent Middleware
- **Lines of Code**: 224
- **Related Classes**: `PlanningState`, `Todo` (TypedDict)

## Overview

`TodoListMiddleware` provides task planning and management capabilities to agents by adding a `write_todos` tool that allows agents to create and manage structured task lists for complex multi-step operations. The middleware helps agents track progress, organize complex tasks, and provide users with visibility into task completion status.

### Purpose
The middleware addresses the challenge of managing complex, multi-step agent operations by providing a structured way for agents to plan work, track progress, and communicate their workflow to users. This is particularly valuable for lengthy operations where understanding current state and remaining work is important.

### Key Features
- **write_todos tool**: Injected tool for creating/updating task lists
- **State tracking**: Maintains task list in agent state with status tracking
- **Guided usage**: Automatic system prompt injection explaining when to use todos
- **Custom prompts**: Configurable tool descriptions and system prompts
- **Status management**: Three-state task model (pending, in_progress, completed)
- **State schema extension**: Extends agent state with `todos` field

## Code Reference

### State Schema Definition

```python
class Todo(TypedDict):
    """A single todo item with content and status."""

    content: str
    """The content/description of the todo item."""

    status: Literal["pending", "in_progress", "completed"]
    """The current status of the todo item."""


class PlanningState(AgentState):
    """State schema for the todo middleware."""

    todos: Annotated[NotRequired[list[Todo]], OmitFromInput]
    """List of todo items for tracking task progress."""
```

**Key Design Points:**
- `Todo`: Simple TypedDict with content and status
- `PlanningState`: Extends `AgentState` with todos field
- `OmitFromInput`: Todos not required in initial input
- `NotRequired`: Field may not be present in all states

### Main Class Definition

```python
class TodoListMiddleware(AgentMiddleware):
    """Middleware that provides todo list management capabilities to agents.

    This middleware adds a `write_todos` tool that allows agents to create and manage
    structured task lists for complex multi-step operations. It's designed to help
    agents track progress, organize complex tasks, and provide users with visibility
    into task completion status.

    The middleware automatically injects system prompts that guide the agent on when
    and how to use the todo functionality effectively.
    """

    state_schema = PlanningState

    def __init__(
        self,
        *,
        system_prompt: str = WRITE_TODOS_SYSTEM_PROMPT,
        tool_description: str = WRITE_TODOS_TOOL_DESCRIPTION,
    ) -> None:
        """Initialize the `TodoListMiddleware` with optional custom prompts.

        Args:
            system_prompt: Custom system prompt to guide the agent on using the todo
                tool.
            tool_description: Custom description for the `write_todos` tool.
        """
        super().__init__()
        self.system_prompt = system_prompt
        self.tool_description = tool_description

        # Dynamically create the write_todos tool with the custom description
        @tool(description=self.tool_description)
        def write_todos(
            todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
        ) -> Command:
            """Create and manage a structured task list for your current work session."""
            return Command(
                update={
                    "todos": todos,
                    "messages": [
                        ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)
                    ],
                }
            )

        self.tools = [write_todos]
```

### System Prompt Injection

```python
def wrap_model_call(
    self,
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelCallResult:
    """Update the system message to include the todo system prompt."""
    if request.system_message is not None:
        new_system_content = [
            *request.system_message.content_blocks,
            {"type": "text", "text": f"\n\n{self.system_prompt}"},
        ]
    else:
        new_system_content = [{"type": "text", "text": self.system_prompt}]
    new_system_message = SystemMessage(
        content=cast("list[str | dict[str, str]]", new_system_content)
    )
    return handler(request.override(system_message=new_system_message))
```

### Default Tool Description

The middleware includes comprehensive guidance (excerpt):

```python
WRITE_TODOS_TOOL_DESCRIPTION = """Use this tool to create and manage a structured task list for your current work session...

## When to Use This Tool
1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests todo list - When the user directly asks you to use the todo list
4. User provides multiple tasks - When users provide a list of things to be done
5. The plan may need future revisions or updates based on results from the first few steps

## When NOT to Use This Tool
1. There is only a single, straightforward task
2. The task is trivial and tracking it provides no benefit
3. The task can be completed in less than 3 trivial steps
4. The task is purely conversational or informational

## Task States and Management
- pending: Task not yet started
- in_progress: Currently working on
- completed: Task finished successfully
..."""
```

### Default System Prompt

```python
WRITE_TODOS_SYSTEM_PROMPT = """## `write_todos`

You have access to the `write_todos` tool to help you manage and plan complex objectives.
Use this tool for complex objectives to ensure that you are tracking each necessary step...

## Important To-Do List Usage Notes to Remember
- The `write_todos` tool should never be called multiple times in parallel.
- Don't be afraid to revise the To-Do list as you go..."""
```

## I/O Contract

### Input
- **ModelRequest**: Standard model request with messages and system prompt
- **Configuration**:
  - `system_prompt`: Custom guidance for using the tool
  - `tool_description`: Custom tool description

### Output
- **Modified Request**: Request with augmented system message
- **write_todos Tool**: Injected into agent's available tools
- **State Updates**: When tool is called, returns `Command` with state updates

### State Schema Extension
The middleware extends agent state with:
```python
todos: list[Todo]  # Optional field for task tracking
```

Each `Todo` contains:
- `content`: String description of the task
- `status`: One of "pending", "in_progress", "completed"

### Tool Execution Flow
1. Agent calls `write_todos` with list of todos
2. Tool returns `Command` updating state
3. State `todos` field updated with new list
4. ToolMessage added confirming update

## Usage Examples

### Basic Usage

```python
from langchain.agents.middleware.todo import TodoListMiddleware
from langchain.agents import create_agent

# Add todo list capability to agent
middleware = TodoListMiddleware()

agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool, file_tool],
    middleware=[middleware],
)

# Agent can now use write_todos tool for complex tasks
result = await agent.invoke({
    "messages": [HumanMessage("Refactor my codebase to use async/await")]
})

# Check task progress
print(result["todos"])
# [
#   {"content": "Analyze current code structure", "status": "completed"},
#   {"content": "Identify sync functions to convert", "status": "completed"},
#   {"content": "Update function signatures", "status": "in_progress"},
#   {"content": "Add await keywords", "status": "pending"},
#   {"content": "Update tests", "status": "pending"},
# ]
```

### Custom Prompts

```python
# Customize guidance for specific use case
custom_system_prompt = """
You have access to the write_todos tool for managing complex data migrations.
Use it to track:
1. Data validation steps
2. Transformation operations
3. Quality checks
4. Rollback procedures

Keep todos focused on concrete, verifiable steps.
"""

custom_tool_description = """
Create a structured task list for data migration work.
Each todo should be a specific, testable operation.
Mark tasks as completed only after verification.
"""

middleware = TodoListMiddleware(
    system_prompt=custom_system_prompt,
    tool_description=custom_tool_description,
)

agent = create_agent(
    model="openai:gpt-4o",
    middleware=[middleware],
)
```

### Monitoring Task Progress

```python
# Stream results to monitor task updates in real-time
async for chunk in agent.astream({
    "messages": [HumanMessage("Build a web scraper")]
}):
    if "todos" in chunk:
        todos = chunk["todos"]
        in_progress = [t for t in todos if t["status"] == "in_progress"]
        completed = [t for t in todos if t["status"] == "completed"]
        print(f"Working on: {in_progress[0]['content'] if in_progress else 'None'}")
        print(f"Completed: {len(completed)}/{len(todos)}")
```

### Multi-Task Request

```python
# User provides multiple tasks - agent automatically uses todos
result = await agent.invoke({
    "messages": [HumanMessage("""
        Please help me with these tasks:
        1. Search for recent AI papers
        2. Summarize the top 3
        3. Create a comparison table
        4. Email the results to team@example.com
    """)]
})

# Agent likely creates todo list to track each task
```

### Without Todo List (Simple Task)

```python
# For simple tasks, agent won't use todo list
result = await agent.invoke({
    "messages": [HumanMessage("What's 2 + 2?")]
})

# result["todos"] likely not present - task too simple
```

### Accessing Todo State

```python
# Access todos from result
result = await agent.invoke({"messages": [HumanMessage("Complex task")]})

if "todos" in result:
    all_todos = result["todos"]
    pending = [t for t in all_todos if t["status"] == "pending"]
    in_progress = [t for t in all_todos if t["status"] == "in_progress"]
    completed = [t for t in all_todos if t["status"] == "completed"]

    print(f"Total tasks: {len(all_todos)}")
    print(f"Pending: {len(pending)}")
    print(f"In progress: {len(in_progress)}")
    print(f"Completed: {len(completed)}")
```

## Implementation Details

### Tool Creation Pattern

The middleware dynamically creates the tool in `__init__`:
```python
@tool(description=self.tool_description)
def write_todos(
    todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Create and manage a structured task list for your current work session."""
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)
            ],
        }
    )
```

This allows custom descriptions while maintaining consistent tool logic.

### System Message Augmentation

The middleware appends to existing system messages:
- Preserves any existing system message content
- Adds todo guidance as additional text block
- Uses content blocks format for compatibility

### State Management

The `todos` field is marked with:
- `OmitFromInput`: Not required in initial input
- `NotRequired`: Optional field in state

This allows:
- Agents to start without todos
- Todos to be added mid-execution
- Clean state when not used

### Command Return Pattern

The tool returns a `Command` object, which is LangGraph's way of:
- Updating state with new todos list
- Adding confirming message to conversation
- Potentially triggering graph control flow

### Tool Call ID Injection

Uses `InjectedToolCallId` for automatic ID injection:
```python
tool_call_id: Annotated[str, InjectedToolCallId]
```

This ensures the ToolMessage has correct `tool_call_id` for message history.

### Async Support

Both sync and async `wrap_model_call` methods implemented:
- `wrap_model_call`: Synchronous version
- `awrap_model_call`: Asynchronous version
- Both apply same system message augmentation

### Guidance Philosophy

The default prompts emphasize:
- **Selective use**: Only for complex tasks
- **Progressive disclosure**: Start simple, add structure as needed
- **Status discipline**: Clear rules for marking completed
- **Revision freedom**: Can update todos as work progresses

## Related Pages

### Core Middleware Infrastructure
- **langchain-ai_langchain_AgentMiddleware_class.md**: Base middleware interface
- **langchain-ai_langchain_middleware_hooks.txt**: Middleware hook system
- **langchain-ai_langchain_middleware_tools.txt**: Tool injection patterns

### Other V1 Middleware Implementations
- **langchain-ai_langchain_ContextEditingMiddleware.md**: Context window management
- **langchain-ai_langchain_ModelCallLimitMiddleware.md**: Model call quota enforcement
- **langchain-ai_langchain_ModelFallbackMiddleware.md**: Model failover on errors
- **langchain-ai_langchain_LLMToolEmulator.md**: LLM-based tool emulation

### Agent Creation
- **langchain-ai_langchain_create_agent.md**: Agent creation with middleware support

### State Management
- **langchain-ai_langchain_state_schema_extension.txt**: State schema extension patterns

### Tool Implementation
- **langchain-ai_langchain_BaseTool_creation.md**: Tool creation patterns

## Architecture Notes

### Design Philosophy
The middleware follows these principles:
- **Opt-in complexity**: Agents only use todos when beneficial
- **User visibility**: Task status visible to end users
- **Flexible planning**: Agents can revise plans as they learn
- **Minimal overhead**: No cost when not used

### Tool Injection Strategy

The middleware adds tools at the middleware level rather than agent creation:
- Tools available automatically with middleware
- No manual tool registration needed
- Tool description customizable per middleware instance

### State Schema Extension Pattern

Demonstrates proper state extension:
1. Define custom state schema extending `AgentState`
2. Add new fields with appropriate annotations
3. Set `state_schema` class attribute on middleware
4. LangGraph merges schemas automatically

### System Prompt Architecture

The middleware augments rather than replaces system prompts:
- Preserves agent's base instructions
- Adds tool-specific guidance
- Uses content blocks for multi-part messages

### Command Pattern Usage

The tool returns `Command` objects for state updates:
- Declarative state changes
- Automatic message history management
- Potential for control flow (jumps, etc.)

### Performance Considerations
- System prompt augmentation happens per model call (lightweight)
- Tool only invoked when agent decides to use it
- State updates only when todos modified
- No background processing or monitoring

### User Experience Design

The middleware improves UX by:
- Making agent planning visible
- Showing progress on long tasks
- Building confidence through transparency
- Allowing understanding of current state

### Multi-Step Workflow Support

Ideal for workflows like:
- Code refactoring projects
- Data migration tasks
- Research and analysis
- Content creation pipelines
- System setup and configuration

## Extension Points

### Custom Todo Schema

```python
class DetailedTodo(TypedDict):
    content: str
    status: Literal["pending", "in_progress", "completed", "blocked"]
    priority: Literal["low", "medium", "high"]
    assignee: str
    estimated_minutes: int

class CustomPlanningState(AgentState):
    todos: Annotated[NotRequired[list[DetailedTodo]], OmitFromInput]

class EnhancedTodoMiddleware(TodoListMiddleware):
    state_schema = CustomPlanningState
    # Override tool creation with enhanced schema
```

### Metrics and Analytics

```python
class MetricsTodoMiddleware(TodoListMiddleware):
    def wrap_model_call(self, request, handler):
        result = super().wrap_model_call(request, handler)

        # Track todo usage
        if "todos" in result:
            self.metrics.record_todo_update(len(result["todos"]))

        return result
```

### Progress Notifications

```python
class NotifyingTodoMiddleware(TodoListMiddleware):
    def __init__(self, *args, notification_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.notify = notification_callback

    def wrap_model_call(self, request, handler):
        result = super().wrap_model_call(request, handler)

        if "todos" in result:
            completed = [t for t in result["todos"] if t["status"] == "completed"]
            if self.notify and completed:
                self.notify(f"{len(completed)} tasks completed")

        return result
```

### Validation

```python
class ValidatedTodoMiddleware(TodoListMiddleware):
    def __init__(self, *args, max_todos=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_todos = max_todos

    def wrap_model_call(self, request, handler):
        # Add validation logic to tool
        original_tool = self.tools[0]

        def validated_write_todos(todos, tool_call_id):
            if len(todos) > self.max_todos:
                return ToolMessage(
                    content=f"Error: Too many todos ({len(todos)}). Maximum is {self.max_todos}.",
                    tool_call_id=tool_call_id,
                )
            return original_tool.invoke({"todos": todos, "tool_call_id": tool_call_id})

        # Replace tool with validated version
        # ... implementation details
```

### Persistence

```python
class PersistentTodoMiddleware(TodoListMiddleware):
    def __init__(self, *args, storage_backend, **kwargs):
        super().__init__(*args, **kwargs)
        self.storage = storage_backend

    def wrap_model_call(self, request, handler):
        result = super().wrap_model_call(request, handler)

        if "todos" in result:
            # Persist todos to external storage
            self.storage.save_todos(
                thread_id=request.config.get("thread_id"),
                todos=result["todos"],
            )

        return result
```
