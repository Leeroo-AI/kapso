# create_agent

**Sources:**
- `libs/langchain_v1/langchain/agents/factory.py:L541-1483`
- LangChain Documentation: Agent Creation

**Domains:** Agent Construction, Graph Building, State Management

**Last Updated:** 2025-12-17

---

## Overview

`create_agent` is a factory function that constructs executable LLM agents by composing models, tools, and middleware into a LangGraph state machine. It automates the complex process of building agent graphs, managing state schemas, composing middleware chains, and configuring control flow.

## Description

The `create_agent` function serves as the primary entry point for creating agentic systems in LangChain. It orchestrates a comprehensive construction process:

1. **Model initialization** - Accepts model strings or instances, uses `init_chat_model` if needed
2. **Tool preparation** - Converts callables to `BaseTool` instances, merges middleware tools
3. **Middleware composition** - Chains middleware hooks, merges state schemas
4. **State graph construction** - Builds nodes for model, tools, and middleware hooks
5. **Control flow definition** - Establishes edges and routing logic for the agent loop
6. **Response format integration** - Configures structured output handling
7. **Graph compilation** - Returns a `CompiledStateGraph` ready for execution

The resulting agent follows a reasoning-action loop:
1. Model generates response (potentially with tool calls)
2. If tool calls exist, execute tools and add results to messages
3. Loop back to model with tool results
4. When no tool calls, exit and return final state

## Code Reference

### Location
**File:** `/tmp/praxium_repo_wjjl6pl8/libs/langchain_v1/langchain/agents/factory.py`
**Lines:** 541-1483

### Signature

```python
def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | None = None,
    state_schema: type[AgentState[ResponseT]] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph[
    AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]
]
```

### Key Internal Components

**Graph Construction:**
```python
# Create StateGraph with merged schemas
graph = StateGraph(
    state_schema=resolved_state_schema,
    input_schema=input_schema,
    output_schema=output_schema,
    context_schema=context_schema,
)

# Add core nodes
graph.add_node("model", model_node)
graph.add_node("tools", tool_node)  # If tools exist

# Add middleware nodes
for middleware in middleware_list:
    graph.add_node(f"{m.name}.before_model", before_model_hook)
    graph.add_node(f"{m.name}.after_model", after_model_hook)
    # ... etc
```

**Control Flow:**
```python
# Entry point
graph.add_edge(START, entry_node)

# Agent loop (model -> tools -> model)
graph.add_conditional_edges(
    "model",
    routing_function,  # Routes to tools or END based on tool_calls
    ["tools", END]
)

graph.add_conditional_edges(
    "tools",
    routing_function,  # Routes back to model or END
    [loop_entry_node, END]
)
```

**Middleware Composition:**
```python
# Chain wrap_model_call handlers
wrap_model_call_handler = _chain_model_call_handlers([
    m1.wrap_model_call,
    m2.wrap_model_call,
    # ... innermost to outermost
])

# Use in model node
response = wrap_model_call_handler(request, _execute_model_sync)
```

## I/O Contract

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str \| BaseChatModel` | Model identifier or instance |
| `tools` | `Sequence[BaseTool \| Callable \| dict] \| None` | Agent capabilities |
| `system_prompt` | `str \| SystemMessage \| None` | System instruction for model |
| `middleware` | `Sequence[AgentMiddleware]` | Middleware for behavior customization |
| `response_format` | `ResponseFormat \| type \| None` | Structured output configuration |
| `state_schema` | `type[AgentState] \| None` | Custom state schema |
| `context_schema` | `type[Context] \| None` | Runtime context schema |
| `checkpointer` | `Checkpointer \| None` | Persistence for conversation memory |
| `store` | `BaseStore \| None` | Cross-thread data storage |
| `interrupt_before` | `list[str] \| None` | Nodes to pause before executing |
| `interrupt_after` | `list[str] \| None` | Nodes to pause after executing |
| `debug` | `bool` | Enable verbose execution logging |
| `name` | `str \| None` | Agent name (used in subgraphs) |
| `cache` | `BaseCache \| None` | Caching for graph execution |

### Outputs

| Type | Description |
|------|-------------|
| `CompiledStateGraph` | Executable agent graph with `invoke()`, `stream()`, `ainvoke()`, `astream()` methods |

### State Schema

**Default AgentState:**
```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    structured_response: Any  # Optional, set when response_format is used
```

**Extended State (with custom schemas):**
Middleware and `state_schema` parameter can add custom fields.

## Usage Examples

### Example 1: Basic Agent with Tools

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

# Create agent
agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_web, calculate],
    system_prompt="You are a helpful assistant with access to web search and a calculator."
)

# Use agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "What is 23 * 47?"}]
})

print(result["messages"][-1].content)
```

### Example 2: Agent with Middleware

```python
from langchain.agents import create_agent, AgentMiddleware, AgentState, Runtime
from typing import Any

class LoggingMiddleware(AgentMiddleware):
    """Log all model calls."""

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"[LOG] Calling model with {len(state['messages'])} messages")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"[LOG] Model returned response")
        return None

agent = create_agent(
    model="gpt-4o",
    tools=[search_web],
    middleware=[LoggingMiddleware()],
    system_prompt="You are a research assistant."
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Research quantum computing"}]
})
```

### Example 3: Agent with Structured Output

```python
from langchain.agents import create_agent
from pydantic import BaseModel, Field

class ResearchReport(BaseModel):
    """A research report summary."""
    topic: str = Field(..., description="Research topic")
    key_findings: list[str] = Field(..., description="Main findings")
    sources: list[str] = Field(..., description="Information sources")

agent = create_agent(
    model="gpt-4o",
    tools=[search_web],
    response_format=ResearchReport,
    system_prompt="Research the given topic and produce a structured report."
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Research renewable energy trends"}]
})

# Access structured output
report: ResearchReport = result["structured_response"]
print(f"Topic: {report.topic}")
print(f"Findings: {report.key_findings}")
```

### Example 4: Agent with Persistence (Checkpointer)

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# Create checkpointer for conversation memory
checkpointer = MemorySaver()

agent = create_agent(
    model="gpt-4o",
    tools=[search_web],
    checkpointer=checkpointer,
    system_prompt="You are a helpful assistant."
)

# First conversation turn
config = {"configurable": {"thread_id": "user_123"}}
result1 = agent.invoke(
    {"messages": [{"role": "user", "content": "My name is Alice"}]},
    config=config
)

# Second turn - agent remembers context
result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    config=config
)

print(result2["messages"][-1].content)  # "Your name is Alice"
```

### Example 5: Agent with Human-in-the-Loop

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def delete_file(filename: str) -> str:
    """Delete a file. Requires approval."""
    return f"Deleted {filename}"

agent = create_agent(
    model="gpt-4o",
    tools=[delete_file],
    interrupt_before=["tools"],  # Pause before tool execution
    system_prompt="You are a file management assistant."
)

# First invocation - stops before tool execution
result1 = agent.invoke({
    "messages": [{"role": "user", "content": "Delete old_data.txt"}]
})

# At this point, inspect the pending tool call
# If approved, resume execution:
result2 = agent.invoke(None, config={"configurable": {"thread_id": "thread_1"}})
```

### Example 6: Streaming Agent Responses

```python
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o",
    tools=[search_web],
    system_prompt="You are a helpful assistant."
)

# Stream updates as they happen
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Research AI safety"}]},
    stream_mode="updates"  # Or "values" for full state
):
    print(chunk)
```

### Example 7: Agent with Custom State Schema

```python
from langchain.agents import create_agent, AgentMiddleware
from typing import TypedDict, Annotated, Any
from langchain_core.messages import AnyMessage, add_messages

class CustomAgentState(TypedDict):
    """Extended state with custom fields."""
    messages: Annotated[list[AnyMessage], add_messages]
    structured_response: Any
    user_context: dict  # Custom field
    iteration_count: int  # Custom field

class ContextMiddleware(AgentMiddleware):
    """Middleware that uses custom state."""

    state_schema = CustomAgentState

    def before_agent(self, state: CustomAgentState, runtime: Runtime) -> dict:
        return {
            "user_context": {"user_id": runtime.context.get("user_id")},
            "iteration_count": 0
        }

    def before_model(self, state: CustomAgentState, runtime: Runtime) -> dict:
        return {"iteration_count": state.get("iteration_count", 0) + 1}

agent = create_agent(
    model="gpt-4o",
    tools=[search_web],
    middleware=[ContextMiddleware()],
    state_schema=CustomAgentState  # Specify custom schema
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Hello"}]
})

print(f"Iterations: {result['iteration_count']}")
print(f"User context: {result['user_context']}")
```

### Example 8: Multi-Model Agent (Configurable)

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

# Create configurable model
configurable_model = init_chat_model(
    "gpt-4o",
    configurable_fields=("model", "temperature")
)

agent = create_agent(
    model=configurable_model,
    tools=[search_web, calculate],
    system_prompt="You are a helpful assistant."
)

# Use with default model (gpt-4o)
result1 = agent.invoke({
    "messages": [{"role": "user", "content": "What is 2+2?"}]
})

# Override to use different model
result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "What is 2+2?"}]},
    config={
        "configurable": {
            "model": "claude-sonnet-4-5-20250929",
            "temperature": 0.7
        }
    }
)
```

### Example 9: Agent with Caching

```python
from langchain.agents import create_agent
from langchain_core.caches import InMemoryCache

cache = InMemoryCache()

agent = create_agent(
    model="gpt-4o",
    tools=[search_web],
    cache=cache,  # Enable caching
    system_prompt="You are a helpful assistant."
)

# First call - executes normally
result1 = agent.invoke({
    "messages": [{"role": "user", "content": "What is Python?"}]
})

# Second call with same input - uses cache
result2 = agent.invoke({
    "messages": [{"role": "user", "content": "What is Python?"}]
})
```

### Example 10: Agent without Tools (Model-Only)

```python
from langchain.agents import create_agent

# Agent with no tools - just model conversation
agent = create_agent(
    model="gpt-4o",
    tools=None,  # Or []
    system_prompt="You are a knowledgeable assistant."
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Explain quantum mechanics"}]
})

# Agent responds but cannot use tools
print(result["messages"][-1].content)
```

## Related Pages

**Principle:**
- [[langchain-ai_langchain_Agent_Graph_Construction]] - Core principle implemented

**Related Implementations:**
- [[langchain-ai_langchain_init_chat_model]] - Model initialization (Step 1)
- [[langchain-ai_langchain_BaseTool_creation]] - Tool creation (Step 2)
- [[langchain-ai_langchain_AgentMiddleware_class]] - Middleware (Step 3)
- [[langchain-ai_langchain_ResponseFormat_strategies]] - Structured output (Step 4)
- [[langchain-ai_langchain_StateGraph]] - LangGraph state graph
- [[langchain-ai_langchain_ToolNode]] - Tool execution node
- [[langchain-ai_langchain_chain_handlers]] - Middleware composition

**Used In:**
- [[langchain-ai_langchain_Agent_Creation_and_Execution]] - create_agent is Step 5

**Workflows:**
- [[langchain-ai_langchain_Agent_Creation_and_Execution]] - End-to-end agent workflow

**Next Step:**
- [[langchain-ai_langchain_CompiledStateGraph_invoke]] - Executing the created agent
