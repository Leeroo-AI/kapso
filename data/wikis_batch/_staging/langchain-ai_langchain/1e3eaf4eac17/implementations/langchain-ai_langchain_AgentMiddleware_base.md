= AgentMiddleware_base =

{{Metadata
|sources=libs/langchain_v1/langchain/agents/middleware/types.py:L330-690
|domains=Agent Architecture, Middleware Implementation, Hook Systems
|last_updated=2025-12-17
}}

== Overview ==

'''AgentMiddleware_base''' provides the concrete implementation of the middleware abstraction through the `AgentMiddleware` base class and a collection of decorator factories. This implementation enables developers to create middleware either by subclassing or by using decorators to convert standalone functions into middleware instances.

== Description ==

The implementation consists of:

'''1. AgentMiddleware Base Class'''

A generic base class with the following structure:
* Generic type parameters: `StateT` (state schema) and `ContextT` (runtime context)
* Class attributes: `state_schema` and `tools`
* Instance property: `name` (defaults to class name)
* Six optional hook methods (sync + async pairs):
** `before_agent` / `abefore_agent`
** `before_model` / `abefore_model`
** `wrap_model_call` / `awrap_model_call`
** `after_model` / `aafter_model`
** `after_agent` / `aafter_agent`
** `wrap_tool_call` / `awrap_tool_call`

'''2. Decorator Factories'''

Five decorator functions that convert standalone functions to middleware:
* `@before_agent` - Converts function to middleware with before_agent hook
* `@before_model` - Converts function to middleware with before_model hook
* `@after_model` - Converts function to middleware with after_model hook
* `@after_agent` - Converts function to middleware with after_agent hook
* `@wrap_model_call` - Converts function to middleware with wrap_model_call hook
* `@wrap_tool_call` - Converts function to middleware with wrap_tool_call hook
* `@dynamic_prompt` - Specialized wrapper for dynamic system prompt generation

'''3. Configuration Helper'''

* `@hook_config` - Decorator to configure hook behavior (e.g., `can_jump_to` destinations)

== Code Reference ==

'''Key Source Locations:'''

<source lang="python">
# Base class definition: lines 330-689
class AgentMiddleware(Generic[StateT, ContextT]):
    state_schema: type[StateT] = cast("type[StateT]", AgentState)
    tools: list[BaseTool]

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run before the agent execution starts."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Intercept and control model execution via handler callback."""
        # Default raises NotImplementedError

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Intercept tool execution for retries, monitoring, or modification."""
        # Default raises NotImplementedError
</source>

'''Decorator Implementation Pattern:'''

<source lang="python">
# Example: before_model decorator (lines 819-950)
def before_model(
    func: _CallableWithStateAndRuntime[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
):
    def decorator(func):
        is_async = iscoroutinefunction(func)

        if is_async:
            async def async_wrapped(_self, state, runtime):
                return await func(state, runtime)

            return type(
                name or func.__name__,
                (AgentMiddleware,),
                {
                    "state_schema": state_schema or AgentState,
                    "tools": tools or [],
                    "abefore_model": async_wrapped,
                },
            )()
        # ... sync implementation

    return decorator(func) if func else decorator
</source>

== I/O Contract ==

'''Base Class Usage:'''

'''Input:'''
* Subclass inherits from `AgentMiddleware[StateT, ContextT]`
* Override one or more hook methods
* Optionally set `state_schema` and `tools` class attributes

'''Output:'''
* Middleware instance that can be passed to `create_agent(middleware=[...])`
* Hook methods return `dict[str, Any]` (state updates), `Command` (flow control), or `None`

'''Decorator Usage:'''

'''Input:'''
* Function with signature matching the hook type
* Optional parameters: `state_schema`, `tools`, `can_jump_to`, `name`

'''Output:'''
* AgentMiddleware instance with the appropriate hook method implemented

'''Type Signatures:'''

<source lang="python">
# Lifecycle hook signature
def hook_func(state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | Command | None

# Model call wrapper signature
def wrap_func(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse | AIMessage

# Tool call wrapper signature
def wrap_func(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command]
) -> ToolMessage | Command
</source>

== Usage Examples ==

'''Example 1: Class-Based Middleware with Multiple Hooks'''

<source lang="python">
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse

class LoggingMiddleware(AgentMiddleware):
    def before_agent(self, state, runtime):
        print(f"Agent starting with {len(state['messages'])} messages")
        return None

    def wrap_model_call(self, request, handler):
        print(f"Calling model: {request.model}")
        response = handler(request)
        print(f"Model returned {len(response.result)} messages")
        return response

    def after_agent(self, state, runtime):
        print(f"Agent completed with {len(state['messages'])} messages")
        return None

# Usage
agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[...],
    middleware=[LoggingMiddleware()]
)
</source>

'''Example 2: Decorator-Based Middleware'''

<source lang="python">
from langchain.agents.middleware.types import before_model, wrap_model_call

@before_model
def validate_inputs(state, runtime):
    """Ensure messages are not empty before calling model."""
    if not state["messages"]:
        return {"messages": [{"role": "user", "content": "Hello"}]}
    return None

@wrap_model_call
def retry_on_error(request, handler):
    """Retry model calls up to 3 times on failure."""
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1} after error: {e}")

# Usage
agent = create_agent(
    model="openai:gpt-4",
    middleware=[validate_inputs, retry_on_error]
)
</source>

'''Example 3: Custom State Schema'''

<source lang="python">
from typing import TypedDict, Annotated
from langchain.agents.middleware.types import AgentMiddleware, AgentState, before_model

class CustomState(AgentState):
    user_id: Annotated[str, OmitFromInput]
    session_count: Annotated[int, PrivateStateAttr]

@before_model(state_schema=CustomState)
def track_sessions(state: CustomState, runtime):
    """Increment session counter for this user."""
    current_count = state.get("session_count", 0)
    return {"session_count": current_count + 1}

# Usage
agent = create_agent(
    model="openai:gpt-4",
    middleware=[track_sessions],
    state_schema=CustomState
)
</source>

'''Example 4: Conditional Jumping'''

<source lang="python">
from langchain.agents.middleware.types import after_model, hook_config

@after_model(can_jump_to=["end", "model"])
def early_exit_on_complete(state, runtime):
    """Exit agent loop if response contains specific marker."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and "[COMPLETE]" in last_message.content:
        return {"jump_to": "end"}
    return None

# Alternative: using @hook_config decorator
class ConditionalMiddleware(AgentMiddleware):
    @hook_config(can_jump_to=["end"])
    def after_model(self, state, runtime):
        if some_condition(state):
            return {"jump_to": "end"}
        return None
</source>

'''Example 5: Dynamic Prompt Generation'''

<source lang="python">
from langchain.agents.middleware.types import dynamic_prompt

@dynamic_prompt
def context_aware_prompt(request):
    """Generate system prompt based on message count."""
    msg_count = len(request.state["messages"])
    user_name = request.runtime.context.get("user_name", "User")

    if msg_count > 10:
        return f"You are assisting {user_name}. Keep responses concise."
    else:
        return f"You are a helpful assistant for {user_name}."

# Usage
agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    middleware=[context_aware_prompt]
)
</source>

'''Example 6: Async Middleware'''

<source lang="python">
from langchain.agents.middleware.types import wrap_model_call

@wrap_model_call
async def async_cache_middleware(request, handler):
    """Cache model responses in an async database."""
    cache_key = hash_request(request)

    # Try to get from cache
    if cached := await async_cache.get(cache_key):
        return ModelResponse(result=[cached], structured_response=None)

    # Call model
    response = await handler(request)

    # Save to cache
    await async_cache.set(cache_key, response.result[0])

    return response
</source>

== Related Pages ==

'''Principle:'''
* [[langchain-ai_langchain_Middleware_Definition]] - Abstract middleware concept

'''Related Implementations:'''
* [[langchain-ai_langchain_middleware_hooks]] - Individual hook implementations
* [[langchain-ai_langchain_state_schema_extension]] - State schema extension mechanism
* [[langchain-ai_langchain_middleware_tools]] - Tool registration mechanism
* [[langchain-ai_langchain_chain_handlers]] - Handler composition logic

'''Related Principles:'''
* [[langchain-ai_langchain_Middleware_Lifecycle_Hooks]] - Hook execution semantics
* [[langchain-ai_langchain_Middleware_Composition_Order]] - Multi-middleware orchestration

[[Category:Implementations]]
[[Category:Agent Architecture]]
[[Category:Middleware Systems]]
