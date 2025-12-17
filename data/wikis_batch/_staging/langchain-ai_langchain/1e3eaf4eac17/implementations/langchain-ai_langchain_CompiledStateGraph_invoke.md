# CompiledStateGraph Invocation

**Sources:**
- LangGraph Documentation (external framework, typically at `../langgraph/`)
- LangGraph API Reference
- Usage patterns in LangChain agent examples

**Domains:** Graph Execution, Runtime Control, Streaming

**Last Updated:** 2025-12-17

---

## Overview

`CompiledStateGraph` invocation methods (`invoke`, `ainvoke`, `stream`, `astream`) execute compiled agent graphs, managing state transitions, handling configuration, and returning results. These methods provide the runtime interface for interacting with agent systems.

## Description

The `CompiledStateGraph` class (from LangGraph) provides four primary invocation methods:

**Synchronous Methods:**
- `invoke(input, config)` - Executes the graph and returns final state
- `stream(input, config, stream_mode)` - Yields intermediate results during execution

**Asynchronous Methods:**
- `ainvoke(input, config)` - Async version of invoke
- `astream(input, config, stream_mode)` - Async version of stream

All methods handle:
- **Input validation** - Ensures input matches graph's input schema
- **State initialization** - Sets up initial state from input
- **Execution orchestration** - Runs nodes in correct order based on edges
- **State persistence** - Saves checkpoints if checkpointer is configured
- **Configuration application** - Applies runtime config (thread_id, callbacks, etc.)
- **Interrupt handling** - Pauses execution at configured interrupt points
- **Result formatting** - Returns state in appropriate format

### Execution Flow

1. **Input Processing**: Convert input dict to graph's state schema
2. **Checkpoint Loading**: If `thread_id` provided, load previous state
3. **Node Execution**: Execute nodes following graph edges
4. **State Updates**: Merge node outputs using state reducers
5. **Checkpoint Saving**: Persist state after each node (if checkpointer exists)
6. **Interrupt Checks**: Pause if current node is in `interrupt_before/after` list
7. **Termination**: Stop when END node is reached or max recursion hit
8. **Output Formatting**: Extract output according to graph's output schema

### Stream Modes

**`"values"` mode:**
Yields complete state after each node execution. Useful for tracking full state evolution.

**`"updates"` mode:**
Yields only the changes (updates) from each node. More efficient when you only care about deltas.

**`"messages"` mode:**
Yields only new messages added to the messages list. Optimized for conversational interfaces.

**`"debug"` mode:**
Yields detailed execution information including node names, timestamps, and errors.

## Code Reference

### Location
**Module:** `langgraph.graph.state` (external dependency)
**Typical Import:** `from langgraph.graph import StateGraph`

### Method Signatures

```python
class CompiledStateGraph:
    """A compiled state graph ready for execution."""

    def invoke(
        self,
        input: dict[str, Any] | None,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """Execute the graph synchronously and return final state.

        Args:
            input: Initial state dict (must match input_schema)
            config: Runtime configuration (thread_id, callbacks, etc.)

        Returns:
            Final state dict (formatted by output_schema)
        """

    async def ainvoke(
        self,
        input: dict[str, Any] | None,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """Execute the graph asynchronously and return final state."""

    def stream(
        self,
        input: dict[str, Any] | None,
        config: RunnableConfig | None = None,
        *,
        stream_mode: str | list[str] = "values",
    ) -> Iterator[dict[str, Any]]:
        """Stream execution synchronously, yielding intermediate results.

        Args:
            input: Initial state dict
            config: Runtime configuration
            stream_mode: "values", "updates", "messages", or "debug"

        Yields:
            State snapshots or updates based on stream_mode
        """

    async def astream(
        self,
        input: dict[str, Any] | None,
        config: RunnableConfig | None = None,
        *,
        stream_mode: str | list[str] = "values",
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream execution asynchronously."""
```

### Configuration Object

```python
class RunnableConfig(TypedDict, total=False):
    """Runtime configuration for graph execution."""

    configurable: dict[str, Any]
    """Runtime-configurable parameters (e.g., thread_id, model overrides)."""

    callbacks: list[BaseCallbackHandler]
    """Callbacks for observability and tracing."""

    recursion_limit: int
    """Maximum number of node executions (default: 25)."""

    max_concurrency: int
    """Maximum parallel node executions."""

    tags: list[str]
    """Tags for categorizing runs."""

    metadata: dict[str, Any]
    """Arbitrary metadata for the run."""
```

## I/O Contract

### Inputs

**`input` parameter:**

| Field | Type | Description |
|-------|------|-------------|
| `messages` | `list[dict \| AnyMessage]` | Conversation history (for agents) |
| Custom fields | `Any` | Additional state fields defined in state_schema |

**`config` parameter:**

| Field | Type | Description |
|-------|------|-------------|
| `configurable.thread_id` | `str` | Thread identifier for persistence |
| `configurable.*` | `Any` | Runtime parameter overrides |
| `callbacks` | `list[BaseCallbackHandler]` | Observability callbacks |
| `recursion_limit` | `int` | Max execution steps |

### Outputs

**`invoke` / `ainvoke` return value:**

| Field | Type | Description |
|-------|------|-------------|
| `messages` | `list[AnyMessage]` | Complete conversation history |
| `structured_response` | `Any` | Structured output (if configured) |
| Custom fields | `Any` | Additional state from middleware |

**`stream` / `astream` yields:**

Depends on `stream_mode`:
- `"values"`: Full state dict after each node
- `"updates"`: State changes dict from each node
- `"messages"`: New messages from each node
- `"debug"`: Execution metadata dict

## Usage Examples

### Example 1: Basic Invocation

```python
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    system_prompt="You are a helpful assistant."
)

# Simple invocation
result = agent.invoke({
    "messages": [{"role": "user", "content": "Hello!"}]
})

print(result["messages"][-1].content)
```

### Example 2: Persistent Conversation

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    checkpointer=checkpointer
)

# Configuration with thread_id
config = {"configurable": {"thread_id": "conversation_123"}}

# Turn 1
result1 = agent.invoke(
    {"messages": [{"role": "user", "content": "My name is Alice"}]},
    config=config
)

# Turn 2 - agent remembers context
result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    config=config
)

print(result2["messages"][-1].content)  # "Your name is Alice"
```

### Example 3: Streaming Execution

```python
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    system_prompt="You are a helpful assistant."
)

# Stream intermediate steps
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Research quantum computing"}]},
    stream_mode="updates"
):
    print(f"Update: {chunk}")
```

### Example 4: Streaming Messages Only

```python
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    system_prompt="You are a helpful assistant."
)

# Stream only new messages
for message_chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Write a poem"}]},
    stream_mode="messages"
):
    if message_chunk:
        print(message_chunk.content, end="", flush=True)
```

### Example 5: Async Execution

```python
from langchain.agents import create_agent
import asyncio

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    system_prompt="You are a helpful assistant."
)

async def main():
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Hello!"}]
    })
    print(result["messages"][-1].content)

asyncio.run(main())
```

### Example 6: Async Streaming

```python
from langchain.agents import create_agent
import asyncio

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    system_prompt="You are a helpful assistant."
)

async def main():
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": "Research AI"}]},
        stream_mode="updates"
    ):
        print(f"Update: {chunk}")

asyncio.run(main())
```

### Example 7: Handling Interrupts

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

agent = create_agent(
    model="gpt-4o",
    tools=[dangerous_tool],
    checkpointer=checkpointer,
    interrupt_before=["tools"]  # Pause before tool execution
)

config = {"configurable": {"thread_id": "session_1"}}

# First invocation - stops at interrupt
result1 = agent.invoke(
    {"messages": [{"role": "user", "content": "Delete all files"}]},
    config=config
)

# Inspect pending tool call
print(f"Pending action: {result1['messages'][-1].tool_calls}")

# Decision: approve or reject
# If approved, resume:
result2 = agent.invoke(None, config=config)

# If rejected, modify state and resume with different instructions
```

### Example 8: With Callbacks for Observability

```python
from langchain.agents import create_agent
from langchain_core.callbacks import StdOutCallbackHandler

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    system_prompt="You are a helpful assistant."
)

# Add callback for tracing
config = {
    "callbacks": [StdOutCallbackHandler()]
}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Research AI"}]},
    config=config
)
```

### Example 9: Recursion Limit Configuration

```python
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    system_prompt="You are a helpful assistant."
)

# Allow more iterations before timeout
config = {
    "recursion_limit": 50  # Default is 25
}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Complex multi-step task"}]},
    config=config
)
```

### Example 10: Multiple Stream Modes

```python
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    system_prompt="You are a helpful assistant."
)

# Stream both values and debug info
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Hello"}]},
    stream_mode=["values", "debug"]
):
    print(chunk)
```

### Example 11: Resume After Interruption

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    checkpointer=checkpointer,
    interrupt_before=["tools"]
)

config = {"configurable": {"thread_id": "thread_1"}}

# Initial invocation - interrupted
result1 = agent.invoke({"messages": [{"role": "user", "content": "Do action"}]}, config=config)

# Resume by passing None as input
result2 = agent.invoke(None, config=config)

# Or resume with additional messages
result3 = agent.invoke(
    {"messages": [{"role": "user", "content": "Actually, cancel that"}]},
    config=config
)
```

### Example 12: Structured Output Retrieval

```python
from langchain.agents import create_agent
from pydantic import BaseModel, Field

class Report(BaseModel):
    """Research report."""
    title: str = Field(..., description="Report title")
    summary: str = Field(..., description="Executive summary")

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    response_format=Report
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Create a report on AI"}]
})

# Access structured output
report: Report = result["structured_response"]
print(f"Title: {report.title}")
print(f"Summary: {report.summary}")
```

## Related Pages

**Principle:**
- [[langchain-ai_langchain_Agent_Execution]] - Core principle of agent execution

**Related Implementations:**
- [[langchain-ai_langchain_create_agent]] - Creates the graph to execute
- [[langchain-ai_langchain_StateGraph]] - LangGraph StateGraph class
- [[langchain-ai_langchain_Checkpointer]] - State persistence
- [[langchain-ai_langchain_RunnableConfig]] - Configuration object

**Used In:**
- [[langchain-ai_langchain_Agent_Creation_and_Execution]] - Execution is Step 6

**Workflows:**
- [[langchain-ai_langchain_Agent_Creation_and_Execution]] - Complete agent workflow

**Related Principles:**
- [[langchain-ai_langchain_Agent_Graph_Construction]] - Creates executable graph
- [[langchain-ai_langchain_Middleware_Configuration]] - Hooks during execution
