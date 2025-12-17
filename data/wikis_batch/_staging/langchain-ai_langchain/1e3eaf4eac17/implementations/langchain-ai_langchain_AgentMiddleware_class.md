# AgentMiddleware Class

**Sources:**
- `libs/langchain_v1/langchain/agents/middleware/types.py:L330-690`
- Example middleware implementations across `libs/partners/`

**Domains:** Agent Middleware, Execution Hooks, State Management

**Last Updated:** 2025-12-17

---

## Overview

`AgentMiddleware` is a base class that defines the contract for extending agent behavior through lifecycle hooks and execution interception. Subclassing `AgentMiddleware` or using decorator functions enables developers to inject custom logic at specific points in the agent execution loop.

## Description

The `AgentMiddleware` class provides a comprehensive set of hooks for customizing agent behavior:

**Lifecycle Hooks:**
- `before_agent(state, runtime)` - Runs once before agent execution starts
- `before_model(state, runtime)` - Runs before each model invocation
- `after_model(state, runtime)` - Runs after each model response
- `after_agent(state, runtime)` - Runs once after agent execution completes

**Execution Interception:**
- `wrap_model_call(request, handler)` - Intercepts and controls model execution
- `wrap_tool_call(request, handler)` - Intercepts and controls tool execution

**Metadata:**
- `state_schema` - TypedDict class defining custom state fields
- `tools` - List of additional tools provided by the middleware
- `name` - Unique identifier for the middleware instance

Each hook comes in sync and async variants (e.g., `before_model` and `abefore_model`), allowing middleware to work in both execution contexts.

### Execution Flow Integration

When an agent executes:
1. All `before_agent` hooks run sequentially
2. Agent loop begins:
   - All `before_model` hooks run
   - Model call passes through `wrap_model_call` chain (innermost to outermost)
   - All `after_model` hooks run
   - If tool calls exist, each passes through `wrap_tool_call` chain
3. Agent loop completes
4. All `after_agent` hooks run

State updates from hooks are merged into agent state using LangGraph's state reducer logic.

## Code Reference

### Location
**File:** `/tmp/praxium_repo_wjjl6pl8/libs/langchain_v1/langchain/agents/middleware/types.py`
**Lines:** 330-690

### Class Definition

```python
class AgentMiddleware(Generic[StateT, ContextT]):
    """Base middleware class for an agent.

    Subclass this and implement any of the defined methods to customize agent behavior
    between steps in the main agent loop.
    """

    state_schema: type[StateT] = cast("type[StateT]", AgentState)
    """The schema for state passed to the middleware nodes."""

    tools: list[BaseTool]
    """Additional tools registered by the middleware."""

    @property
    def name(self) -> str:
        """The name of the middleware instance."""
        return self.__class__.__name__

    # Lifecycle hooks (sync)
    def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run before the agent execution starts."""

    def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run before the model is called."""

    def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run after the model is called."""

    def after_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run after the agent execution completes."""

    # Lifecycle hooks (async)
    async def abefore_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Async logic to run before the agent execution starts."""

    async def abefore_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Async logic to run before the model is called."""

    async def aafter_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Async logic to run after the model is called."""

    async def aafter_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Async logic to run after the agent execution completes."""

    # Execution interception (sync)
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Intercept and control model execution via handler callback."""
        raise NotImplementedError  # Default raises helpful error

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Intercept tool execution for retries, monitoring, or modification."""
        raise NotImplementedError  # Default raises helpful error

    # Execution interception (async)
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Intercept and control async model execution via handler callback."""
        raise NotImplementedError  # Default raises helpful error

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Intercept and control async tool execution via handler callback."""
        raise NotImplementedError  # Default raises helpful error
```

### Key Types

```python
@dataclass
class ModelRequest:
    """Request object passed to wrap_model_call."""
    model: BaseChatModel
    tools: list[BaseTool]
    system_message: SystemMessage | None
    response_format: ResponseFormat | None
    messages: list[AnyMessage]
    tool_choice: str | None
    model_settings: dict[str, Any]
    state: AgentState
    runtime: Runtime

@dataclass
class ModelResponse:
    """Response from model execution."""
    result: list[AIMessage]
    structured_response: Any | None

@dataclass
class ToolCallRequest:
    """Request object passed to wrap_tool_call."""
    tool_call: dict[str, Any]  # {"name": str, "args": dict, "id": str}
    tool: BaseTool
    state: AgentState
    runtime: Runtime
```

## I/O Contract

### Lifecycle Hooks

**Inputs:**
- `state: StateT` - Current agent state (includes messages, structured_response, custom fields)
- `runtime: Runtime[ContextT]` - Runtime context (user_id, thread_id, checkpointer, store)

**Outputs:**
- `dict[str, Any] | None` - State updates to merge, or `None` for no updates

### Execution Interception

**Model Call Interception:**

**Inputs:**
- `request: ModelRequest` - Contains model, tools, messages, state, runtime
- `handler: Callable` - Function to invoke the model (can call multiple times or skip)

**Outputs:**
- `ModelResponse` - Model result with messages and optional structured_response
- Can also return `AIMessage` directly (auto-converted to `ModelResponse`)

**Tool Call Interception:**

**Inputs:**
- `request: ToolCallRequest` - Contains tool_call dict, BaseTool instance, state, runtime
- `handler: Callable` - Function to execute the tool (can call multiple times or skip)

**Outputs:**
- `ToolMessage` - Result of tool execution
- `Command` - LangGraph command for advanced control flow

## Usage Examples

### Example 1: Simple Logging Middleware

```python
from langchain.agents import AgentMiddleware, AgentState, Runtime
from typing import Any

class LoggingMiddleware(AgentMiddleware):
    """Log agent execution at key points."""

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Log before model calls."""
        print(f"[Before Model] Messages: {len(state['messages'])}")
        return None  # No state updates

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Log after model calls."""
        last_message = state["messages"][-1]
        print(f"[After Model] Response: {last_message.content[:100]}")
        return None

# Use in agent
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[LoggingMiddleware()]
)
```

### Example 2: Retry Middleware with wrap_model_call

```python
from langchain.agents import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
import time

class RetryMiddleware(AgentMiddleware):
    """Retry model calls on failure."""

    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        self.max_retries = max_retries
        self.delay = delay

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Retry model calls with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                return handler(request)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise  # Final attempt failed
                print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(self.delay * (2 ** attempt))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async retry logic."""
        import asyncio
        for attempt in range(self.max_retries):
            try:
                return await handler(request)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                await asyncio.sleep(self.delay * (2 ** attempt))

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[RetryMiddleware(max_retries=3, delay=1.0)]
)
```

### Example 3: Caching Middleware

```python
from langchain.agents import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from typing import Dict, Tuple
import hashlib
import json

class CacheMiddleware(AgentMiddleware):
    """Cache model responses based on message content."""

    def __init__(self):
        self.cache: Dict[str, ModelResponse] = {}

    def _get_cache_key(self, request: ModelRequest) -> str:
        """Generate cache key from messages."""
        content = json.dumps([msg.content for msg in request.messages], sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Check cache before calling model."""
        cache_key = self._get_cache_key(request)

        if cache_key in self.cache:
            print(f"[Cache Hit] Returning cached response")
            return self.cache[cache_key]

        print(f"[Cache Miss] Calling model")
        response = handler(request)
        self.cache[cache_key] = response
        return response

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[CacheMiddleware()]
)
```

### Example 4: Context Injection Middleware (RAG)

```python
from langchain.agents import AgentMiddleware, AgentState, Runtime
from typing import Any

class RAGMiddleware(AgentMiddleware):
    """Inject relevant context before model calls."""

    def __init__(self, retriever):
        self.retriever = retriever

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Retrieve and inject context."""
        last_user_message = next(
            (msg for msg in reversed(state["messages"]) if msg.type == "human"),
            None
        )

        if last_user_message:
            # Retrieve relevant documents
            docs = self.retriever.get_relevant_documents(last_user_message.content)
            context = "\n\n".join([doc.page_content for doc in docs])

            # Inject as system message
            from langchain_core.messages import SystemMessage
            context_message = SystemMessage(
                content=f"Relevant context:\n{context}"
            )

            # Return state update to inject message
            return {"messages": [context_message]}

        return None

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Setup retriever
vectorstore = FAISS.from_texts([...], OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[RAGMiddleware(retriever)]
)
```

### Example 5: State Extension with Custom Schema

```python
from langchain.agents import AgentMiddleware, AgentState, Runtime
from typing import TypedDict, Annotated, Any
from langchain_core.messages import AnyMessage

class MyStateSchema(TypedDict):
    """Custom state with additional fields."""
    messages: Annotated[list[AnyMessage], "messages reducer"]
    structured_response: Any
    user_profile: dict  # Custom field
    query_count: int  # Custom field

class UserProfileMiddleware(AgentMiddleware):
    """Track user profile and query count."""

    state_schema = MyStateSchema  # Declare custom state

    def before_agent(self, state: MyStateSchema, runtime: Runtime) -> dict[str, Any] | None:
        """Initialize user profile from runtime."""
        user_id = runtime.context.get("user_id")

        # Load user profile (simplified)
        profile = {"user_id": user_id, "preferences": {}}

        return {
            "user_profile": profile,
            "query_count": 0
        }

    def before_model(self, state: MyStateSchema, runtime: Runtime) -> dict[str, Any] | None:
        """Increment query count."""
        return {"query_count": state.get("query_count", 0) + 1}

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[UserProfileMiddleware()]
)

# State now has user_profile and query_count fields
result = agent.invoke({"messages": [...]})
print(result["query_count"])  # Access custom state
```

### Example 6: Tool Validation Middleware

```python
from langchain.agents import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage

class ToolValidationMiddleware(AgentMiddleware):
    """Validate tool arguments before execution."""

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        """Validate arguments before calling tool."""
        tool_name = request.tool_call["name"]
        args = request.tool_call["args"]

        # Custom validation logic
        if tool_name == "database_query" and len(args.get("query", "")) > 1000:
            # Return error instead of executing
            return ToolMessage(
                content="Error: Query too long (max 1000 chars)",
                tool_call_id=request.tool_call["id"]
            )

        # Validation passed, execute tool
        return handler(request)

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[ToolValidationMiddleware()]
)
```

### Example 7: Middleware with Custom Tools

```python
from langchain.agents import AgentMiddleware
from langchain_core.tools import tool

class MemoryMiddleware(AgentMiddleware):
    """Middleware that provides memory-related tools."""

    def __init__(self):
        # Define tool
        @tool
        def recall_memory(query: str) -> str:
            """Recall information from long-term memory."""
            # Simplified memory lookup
            return f"Recalled: {query}"

        # Register tools with middleware
        self.tools = [recall_memory]

# Tools are automatically made available to agent
agent = create_agent(
    model="gpt-4o",
    tools=[other_tool],  # Explicit tools
    middleware=[MemoryMiddleware()]  # Middleware tools added automatically
)
```

## Related Pages

**Principle:**
- [[langchain-ai_langchain_Middleware_Configuration]] - Core principle implemented by this class

**Related Implementations:**
- [[langchain-ai_langchain_wrap_model_call]] - Model call interception hook
- [[langchain-ai_langchain_wrap_tool_call]] - Tool call interception hook
- [[langchain-ai_langchain_chain_handlers]] - Middleware composition logic
- [[langchain-ai_langchain_Runtime]] - Runtime context object

**Used In:**
- [[langchain-ai_langchain_create_agent]] - Accepts middleware list parameter
- [[langchain-ai_langchain_Agent_Creation_and_Execution]] - Middleware is Step 3

**Workflows:**
- [[langchain-ai_langchain_Middleware_Composition]] - Detailed middleware workflow
- [[langchain-ai_langchain_Agent_Creation_and_Execution]] - Uses middleware in agent creation
