# ModelCallLimitMiddleware Implementation

## Metadata
- **Component**: `ModelCallLimitMiddleware`
- **Package**: `langchain.agents.middleware.model_call_limit`
- **File Path**: `/tmp/praxium_repo_wjjl6pl8/libs/langchain_v1/langchain/agents/middleware/model_call_limit.py`
- **Type**: Agent Middleware
- **Lines of Code**: 256
- **Related Classes**: `ModelCallLimitState`, `ModelCallLimitExceededError`

## Overview

`ModelCallLimitMiddleware` is a monitoring and enforcement middleware that tracks the number of model calls made during agent execution and can terminate the agent when specified limits are reached. It provides both thread-level (persistent across runs) and run-level (single invocation) call counting with configurable exit behaviors.

### Purpose
The middleware addresses resource management and cost control concerns by preventing runaway agent loops and enforcing maximum model call budgets. This is critical for production deployments where unbounded model usage can lead to unexpected costs or performance issues.

### Key Features
- **Dual-level tracking**: Separate counters for thread-level and run-level limits
- **Persistent thread state**: Thread call counts persist across multiple invocations
- **Configurable exit behavior**: Choose between graceful termination or exception raising
- **Automatic injection**: Injects artificial AI message when limits exceeded
- **Runtime integration**: Uses LangGraph runtime hooks for state management

## Code Reference

### State Schema Extension

```python
class ModelCallLimitState(AgentState):
    """State schema for `ModelCallLimitMiddleware`.

    Extends `AgentState` with model call tracking fields.
    """

    thread_model_call_count: NotRequired[Annotated[int, PrivateStateAttr]]
    run_model_call_count: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]
```

**Key Design Points:**
- `thread_model_call_count`: Persisted across runs, tracked in thread state
- `run_model_call_count`: Marked as `UntrackedValue` so it doesn't persist between runs
- Both marked as `PrivateStateAttr` to keep them internal to middleware

### Main Class Definition

```python
class ModelCallLimitMiddleware(AgentMiddleware[ModelCallLimitState, Any]):
    """Tracks model call counts and enforces limits.

    This middleware monitors the number of model calls made during agent execution
    and can terminate the agent when specified limits are reached. It supports
    both thread-level and run-level call counting with configurable exit behaviors.

    Thread-level: The middleware tracks the number of model calls and persists
    call count across multiple runs (invocations) of the agent.

    Run-level: The middleware tracks the number of model calls made during a single
    run (invocation) of the agent.
    """

    state_schema = ModelCallLimitState

    def __init__(
        self,
        *,
        thread_limit: int | None = None,
        run_limit: int | None = None,
        exit_behavior: Literal["end", "error"] = "end",
    ) -> None:
        """Initialize the call tracking middleware.

        Args:
            thread_limit: Maximum number of model calls allowed per thread.
                `None` means no limit.
            run_limit: Maximum number of model calls allowed per run.
                `None` means no limit.
            exit_behavior: What to do when limits are exceeded.
                - `'end'`: Jump to the end of the agent execution and
                    inject an artificial AI message indicating that the limit was
                    exceeded.
                - `'error'`: Raise a `ModelCallLimitExceededError`

        Raises:
            ValueError: If both limits are `None` or if `exit_behavior` is invalid.
        """
        super().__init__()

        if thread_limit is None and run_limit is None:
            msg = "At least one limit must be specified (thread_limit or run_limit)"
            raise ValueError(msg)

        if exit_behavior not in ("end", "error"):
            msg = f"Invalid exit_behavior: {exit_behavior}. Must be 'end' or 'error'"
            raise ValueError(msg)

        self.thread_limit = thread_limit
        self.run_limit = run_limit
        self.exit_behavior = exit_behavior
```

### Before Model Hook

```python
@hook_config(can_jump_to=["end"])
@override
def before_model(self, state: ModelCallLimitState, runtime: Runtime) -> dict[str, Any] | None:
    """Check model call limits before making a model call.

    Args:
        state: The current agent state containing call counts.
        runtime: The langgraph runtime.

    Returns:
        If limits are exceeded and exit_behavior is `'end'`, returns
            a `Command` to jump to the end with a limit exceeded message. Otherwise
            returns `None`.

    Raises:
        ModelCallLimitExceededError: If limits are exceeded and `exit_behavior`
            is `'error'`.
    """
    thread_count = state.get("thread_model_call_count", 0)
    run_count = state.get("run_model_call_count", 0)

    # Check if any limits will be exceeded after the next call
    thread_limit_exceeded = self.thread_limit is not None and thread_count >= self.thread_limit
    run_limit_exceeded = self.run_limit is not None and run_count >= self.run_limit

    if thread_limit_exceeded or run_limit_exceeded:
        if self.exit_behavior == "error":
            raise ModelCallLimitExceededError(
                thread_count=thread_count,
                run_count=run_count,
                thread_limit=self.thread_limit,
                run_limit=self.run_limit,
            )
        if self.exit_behavior == "end":
            # Create a message indicating the limit was exceeded
            limit_message = _build_limit_exceeded_message(
                thread_count, run_count, self.thread_limit, self.run_limit
            )
            limit_ai_message = AIMessage(content=limit_message)

            return {"jump_to": "end", "messages": [limit_ai_message]}

    return None
```

### After Model Hook

```python
@override
def after_model(self, state: ModelCallLimitState, runtime: Runtime) -> dict[str, Any] | None:
    """Increment model call counts after a model call.

    Args:
        state: The current agent state.
        runtime: The langgraph runtime.

    Returns:
        State updates with incremented call counts.
    """
    return {
        "thread_model_call_count": state.get("thread_model_call_count", 0) + 1,
        "run_model_call_count": state.get("run_model_call_count", 0) + 1,
    }
```

### Exception Class

```python
class ModelCallLimitExceededError(Exception):
    """Exception raised when model call limits are exceeded.

    This exception is raised when the configured exit behavior is `'error'` and either
    the thread or run model call limit has been exceeded.
    """

    def __init__(
        self,
        thread_count: int,
        run_count: int,
        thread_limit: int | None,
        run_limit: int | None,
    ) -> None:
        """Initialize the exception with call count information."""
        self.thread_count = thread_count
        self.run_count = run_count
        self.thread_limit = thread_limit
        self.run_limit = run_limit

        msg = _build_limit_exceeded_message(thread_count, run_count, thread_limit, run_limit)
        super().__init__(msg)
```

## I/O Contract

### Input
- **State Fields**:
  - `thread_model_call_count`: Current thread-level call count (persistent)
  - `run_model_call_count`: Current run-level call count (ephemeral)
- **Configuration**:
  - `thread_limit`: Maximum calls per thread (optional)
  - `run_limit`: Maximum calls per run (optional)
  - `exit_behavior`: How to handle limit violations

### Output
- **State Updates**: Incremented call counters after each model call
- **Jump Commands**: When `exit_behavior="end"`, returns command to jump to end
- **Exceptions**: When `exit_behavior="error"`, raises `ModelCallLimitExceededError`
- **AI Messages**: Injects message explaining limit exceeded

### Hook Execution Order
1. **before_model**: Check if limits will be exceeded, potentially terminate
2. **Model Call**: If limits not exceeded, model is invoked
3. **after_model**: Increment both counters regardless of model call success

## Usage Examples

### Basic Usage with Run Limit

```python
from langchain.agents.middleware.model_call_limit import ModelCallLimitMiddleware
from langchain.agents import create_agent

# Create middleware with run-level limit
call_limiter = ModelCallLimitMiddleware(
    run_limit=5,           # Maximum 5 calls per invocation
    exit_behavior="end"    # Gracefully end when exceeded
)

agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool, calculator_tool],
    middleware=[call_limiter],
)

# Agent will stop after 5 model calls and inject completion message
result = await agent.invoke({"messages": [HumanMessage("Complex task")]})
```

### Thread-Level Budget Control

```python
# Enforce thread-level budget for long conversations
thread_limiter = ModelCallLimitMiddleware(
    thread_limit=50,       # Maximum 50 calls total in this thread
    exit_behavior="end"
)

agent = create_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=[web_search, file_manager],
    middleware=[thread_limiter],
)

# First invocation - uses some of budget
result1 = await agent.invoke(
    {"messages": [HumanMessage("Task 1")]},
    config={"configurable": {"thread_id": "user-123"}}
)

# Later invocation - continues counting from previous
result2 = await agent.invoke(
    {"messages": [HumanMessage("Task 2")]},
    config={"configurable": {"thread_id": "user-123"}}
)
```

### Combined Limits

```python
# Use both thread and run limits together
combined_limiter = ModelCallLimitMiddleware(
    thread_limit=100,      # Max 100 calls lifetime
    run_limit=10,          # Max 10 calls per invocation
    exit_behavior="end"
)

agent = create_agent(
    model="openai:gpt-4o",
    middleware=[combined_limiter],
)

# Will stop at whichever limit is hit first
```

### Error-Based Exit

```python
from langchain.agents.middleware.model_call_limit import (
    ModelCallLimitMiddleware,
    ModelCallLimitExceededError,
)

# Raise exception instead of graceful exit
strict_limiter = ModelCallLimitMiddleware(
    run_limit=3,
    exit_behavior="error"  # Raise exception when exceeded
)

agent = create_agent(
    model="openai:gpt-4o",
    middleware=[strict_limiter],
)

try:
    result = await agent.invoke({"messages": [HumanMessage("Task")]})
except ModelCallLimitExceededError as e:
    print(f"Limit exceeded: thread={e.thread_count}, run={e.run_count}")
    print(f"Limits were: thread={e.thread_limit}, run={e.run_limit}")
```

### Cost Control in Production

```python
# Implement per-user budget control
def create_user_agent(user_id: str, monthly_limit: int):
    limiter = ModelCallLimitMiddleware(
        thread_limit=monthly_limit,
        exit_behavior="end"
    )

    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[search_tool],
        middleware=[limiter],
    )

    return agent

# Each user gets their own budget
agent = create_user_agent("user-123", monthly_limit=1000)
result = await agent.invoke(
    {"messages": [HumanMessage("Help me")]},
    config={"configurable": {"thread_id": f"user-123-december"}}
)
```

### Accessing Call Counts

```python
# Check call counts in returned state
result = await agent.invoke({"messages": [HumanMessage("Task")]})

# Note: these are private state attrs, may not be directly accessible
# Implementation may require custom state inspection
```

## Implementation Details

### State Persistence Strategy

The middleware uses LangGraph's state annotations to control persistence:

1. **Thread Counter** (`thread_model_call_count`):
   - Marked as `PrivateStateAttr` only
   - Persists in thread state between invocations
   - Increments monotonically over thread lifetime

2. **Run Counter** (`run_model_call_count`):
   - Marked as both `UntrackedValue` and `PrivateStateAttr`
   - `UntrackedValue` prevents persistence between runs
   - Resets to 0 for each new invocation
   - Still accessible within single run

### Limit Checking Logic

The `before_model` hook checks limits **before** incrementing:
```python
thread_limit_exceeded = self.thread_limit is not None and thread_count >= self.thread_limit
run_limit_exceeded = self.run_limit is not None and run_count >= self.run_limit
```

This means:
- With `run_limit=5`, the 6th call is prevented (not the 5th)
- The check happens before the model call, preventing wasted API calls

### Exit Behaviors

**End Behavior** (`exit_behavior="end"`):
1. Uses `@hook_config(can_jump_to=["end"])` decorator
2. Returns dict with `jump_to="end"` to trigger graph jump
3. Injects `AIMessage` explaining limit exceeded
4. Agent completes gracefully with final message

**Error Behavior** (`exit_behavior="error"`):
1. Raises `ModelCallLimitExceededError` immediately
2. Exception propagates to caller
3. No graceful completion
4. Allows custom error handling

### Message Format

When limits exceeded with `exit_behavior="end"`, injected message format:
```
Model call limits exceeded: thread limit (50/50)
Model call limits exceeded: run limit (5/5)
Model call limits exceeded: thread limit (100/100), run limit (10/10)
```

Built by `_build_limit_exceeded_message` helper function.

### Async Support

Both sync and async versions implemented:
- `before_model` / `abefore_model`
- `after_model` / `aafter_model`

Async versions delegate to sync implementations for this middleware.

## Related Pages

### Core Middleware Infrastructure
- **langchain-ai_langchain_AgentMiddleware_class.md**: Base middleware interface
- **langchain-ai_langchain_middleware_hooks.txt**: Middleware hook system documentation

### Other V1 Middleware Implementations
- **langchain-ai_langchain_ContextEditingMiddleware.md**: Context window management
- **langchain-ai_langchain_ModelFallbackMiddleware.md**: Model failover on errors
- **langchain-ai_langchain_TodoListMiddleware.md**: Task tracking for agents
- **langchain-ai_langchain_LLMToolEmulator.md**: LLM-based tool emulation

### Related Middleware
- **langchain-ai_langchain_ToolCallLimitMiddleware.md**: Similar pattern for tool calls
- **langchain-ai_langchain_ModelRetryMiddleware.md**: Retry logic for models

### Agent Creation
- **langchain-ai_langchain_create_agent.md**: Agent creation with middleware support

### State Management
- **langchain-ai_langchain_state_schema_extension.txt**: State schema patterns

## Architecture Notes

### Design Philosophy
The middleware follows these principles:
- **Non-invasive**: Uses hooks to avoid wrapping model calls
- **Stateful**: Leverages LangGraph state management for persistence
- **Configurable**: Multiple options for different use cases
- **Transparent**: Clear messaging when limits exceeded

### Thread vs Run Semantics

**Thread-level** is for:
- Long-term budget control
- Multi-session conversations
- Per-user quotas
- Monthly/weekly limits

**Run-level** is for:
- Single invocation constraints
- Preventing runaway loops
- Individual task budgets
- Testing and development

### Performance Considerations
- Counter increments are lightweight (O(1))
- No external storage or API calls
- State updates happen synchronously
- Minimal overhead per model call

### Private State Attributes

Both counters use `PrivateStateAttr`:
- Not exposed in agent's public input schema
- Internal to middleware operation
- May require special access patterns to inspect

### Jump Command Integration

The `@hook_config(can_jump_to=["end"])` decorator:
- Declares middleware can trigger graph jumps
- Enables LangGraph to properly route control flow
- Required for `jump_to` functionality to work

## Error Handling Patterns

### Graceful Degradation
```python
# Agent completes with explanatory message
limiter = ModelCallLimitMiddleware(run_limit=5, exit_behavior="end")
result = await agent.invoke(...)
final_message = result["messages"][-1].content
# "Model call limits exceeded: run limit (5/5)"
```

### Fail-Fast
```python
# Application code handles limit as error
limiter = ModelCallLimitMiddleware(run_limit=5, exit_behavior="error")
try:
    result = await agent.invoke(...)
except ModelCallLimitExceededError as e:
    log_quota_exceeded(user_id, e.thread_count, e.run_count)
    notify_user_quota_limit()
```

### Validation
```python
# Middleware validates configuration at initialization
try:
    bad_limiter = ModelCallLimitMiddleware()  # No limits specified
except ValueError as e:
    # "At least one limit must be specified"
    pass
```

## Extension Points

### Custom Limit Logic
Extend the middleware to implement custom limit checking:
```python
class CustomLimitMiddleware(ModelCallLimitMiddleware):
    def before_model(self, state, runtime):
        # Custom logic here
        # e.g., time-based limits, cost-based limits, etc.
        return super().before_model(state, runtime)
```

### Custom Messages
Override message generation:
```python
def custom_limit_message(thread_count, run_count, thread_limit, run_limit):
    return f"Budget exhausted. Used {run_count} calls this run."

# Use in custom middleware subclass
```

### Integration with Billing
```python
class BillingAwareLimitMiddleware(ModelCallLimitMiddleware):
    def after_model(self, state, runtime):
        result = super().after_model(state, runtime)
        # Log to billing system
        self.billing_tracker.record_model_call(state["thread_id"])
        return result
```
