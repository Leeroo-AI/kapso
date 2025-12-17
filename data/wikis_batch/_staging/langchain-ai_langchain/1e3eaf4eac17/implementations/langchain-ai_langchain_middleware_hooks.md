= middleware_hooks =

{{Metadata
|sources=libs/langchain_v1/langchain/agents/middleware/types.py:L351-690, libs/langchain_v1/langchain/agents/factory.py:L1116-1196
|domains=Agent Execution, Hook Implementation, Middleware Runtime
|last_updated=2025-12-17
}}

== Overview ==

'''middleware_hooks''' provides the concrete implementation of the six lifecycle hook methods within the AgentMiddleware class and the runtime logic in the agent factory that invokes these hooks at the appropriate times. This implementation translates the abstract hook concept into executable code that integrates with the LangGraph-based agent execution engine.

== Description ==

The implementation spans two key areas:

'''1. Hook Method Definitions (types.py)'''

Each hook is defined as a method on the AgentMiddleware base class with:
* Sync version (e.g., `before_agent`)
* Async version (e.g., `abefore_agent`)
* Default implementation (either no-op or NotImplementedError)
* Type annotations for parameters and return values
* Comprehensive docstrings with usage examples

'''2. Hook Invocation Runtime (factory.py)'''

The `create_agent` function integrates hooks into the LangGraph StateGraph:
* Detects which middleware implements which hooks
* Creates graph nodes for each hook instance
* Wires hooks into the execution flow
* Composes multiple middleware instances correctly
* Selects sync vs. async implementation based on invocation

'''Hook Categories:'''

'''Lifecycle Hooks''' (before_agent, after_agent, before_model, after_model):
* Accept `state` and `runtime` parameters
* Return `dict[str, Any]` (state updates), `Command` (flow control), or `None`
* Implemented as graph nodes using RunnableCallable
* Connected via edges or conditional edges (if `can_jump_to` configured)

'''Wrapper Hooks''' (wrap_model_call, wrap_tool_call):
* Accept `request` and `handler` callback parameters
* Must call `handler(request)` to proceed with execution
* Can call handler multiple times (retry) or skip it (short-circuit)
* Return the final result (ModelResponse/AIMessage or ToolMessage/Command)
* Composed using `_chain_model_call_handlers` and `_chain_tool_call_wrappers`

== Code Reference ==

'''Hook Method Signatures:'''

<source lang="python">
# Lifecycle hook signatures (lines 351-537)
class AgentMiddleware(Generic[StateT, ContextT]):
    def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run before the agent execution starts."""

    async def abefore_agent(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async logic to run before the agent execution starts."""

    def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run before the model is called."""

    async def abefore_model(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async logic to run before the model is called."""

    def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run after the model is called."""

    async def aafter_model(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async logic to run after the model is called."""

    def after_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run after the agent execution completes."""

    async def aafter_agent(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async logic to run after the agent execution completes."""
</source>

'''Wrapper Hook Signatures:'''

<source lang="python">
# Wrapper hook signatures (lines 384-688)
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Intercept and control model execution via handler callback."""
        msg = "Synchronous implementation of wrap_model_call is not available..."
        raise NotImplementedError(msg)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Intercept and control async model execution via handler callback."""
        msg = "Asynchronous implementation of awrap_model_call is not available..."
        raise NotImplementedError(msg)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Intercept tool execution for retries, monitoring, or modification."""
        msg = "Synchronous implementation of wrap_tool_call is not available..."
        raise NotImplementedError(msg)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Intercept and control async tool execution via handler callback."""
        msg = "Asynchronous implementation of awrap_tool_call is not available..."
        raise NotImplementedError(msg)
</source>

'''Runtime Hook Integration:'''

<source lang="python">
# Model node with wrap_model_call integration (factory.py lines 1116-1196)
def model_node(state: AgentState, runtime: Runtime[ContextT]) -> dict[str, Any]:
    """Sync model request handler with sequential middleware processing."""
    request = ModelRequest(
        model=model,
        tools=default_tools,
        system_message=system_message,
        response_format=initial_response_format,
        messages=state["messages"],
        tool_choice=None,
        state=state,
        runtime=runtime,
    )

    if wrap_model_call_handler is None:
        # No handlers - execute directly
        response = _execute_model_sync(request)
    else:
        # Call composed handler with base handler
        response = wrap_model_call_handler(request, _execute_model_sync)

    # Extract state updates from ModelResponse
    state_updates = {"messages": response.result}
    if response.structured_response is not None:
        state_updates["structured_response"] = response.structured_response

    return state_updates

async def amodel_node(state: AgentState, runtime: Runtime[ContextT]) -> dict[str, Any]:
    """Async model request handler with sequential middleware processing."""
    request = ModelRequest(...)

    if awrap_model_call_handler is None:
        response = await _execute_model_async(request)
    else:
        response = await awrap_model_call_handler(request, _execute_model_async)

    return state_updates
</source>

'''Hook Detection and Graph Node Creation:'''

<source lang="python">
# Detecting middleware with hooks (factory.py lines 797-838)
middleware_w_before_agent = [
    m for m in middleware
    if m.__class__.before_agent is not AgentMiddleware.before_agent
    or m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
]

middleware_w_before_model = [
    m for m in middleware
    if m.__class__.before_model is not AgentMiddleware.before_model
    or m.__class__.abefore_model is not AgentMiddleware.abefore_model
]

# Creating graph nodes for hooks (factory.py lines 1207-1287)
for m in middleware:
    if (m.__class__.before_agent is not AgentMiddleware.before_agent
        or m.__class__.abefore_agent is not AgentMiddleware.abefore_agent):

        sync_before_agent = (
            m.before_agent
            if m.__class__.before_agent is not AgentMiddleware.before_agent
            else None
        )
        async_before_agent = (
            m.abefore_agent
            if m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
            else None
        )
        before_agent_node = RunnableCallable(sync_before_agent, async_before_agent, trace=False)
        graph.add_node(
            f"{m.name}.before_agent", before_agent_node, input_schema=resolved_state_schema
        )
</source>

== I/O Contract ==

'''Lifecycle Hook Contract:'''

'''Input:'''
* `state: StateT` - Current agent state (includes messages, custom fields)
* `runtime: Runtime[ContextT]` - Runtime context with threading, config, etc.

'''Output:'''
* `dict[str, Any]` - State updates to merge (e.g., `{"custom_field": value}`)
* `Command` - Flow control command (e.g., `Command(goto="end")`)
* `None` - No state changes

'''Wrapper Hook Contract:'''

'''Input (wrap_model_call):'''
* `request: ModelRequest` - Contains model, messages, tools, state, runtime
* `handler: Callable[[ModelRequest], ModelResponse]` - Function to execute model

'''Output (wrap_model_call):'''
* `ModelResponse` - Full response with result messages and structured_response
* `AIMessage` - Simplified return (automatically wrapped in ModelResponse)

'''Input (wrap_tool_call):'''
* `request: ToolCallRequest` - Contains tool_call dict, tool instance, state, runtime
* `handler: Callable[[ToolCallRequest], ToolMessage | Command]` - Function to execute tool

'''Output (wrap_tool_call):'''
* `ToolMessage` - Tool execution result
* `Command` - Flow control command

'''State Update Semantics:'''
* Returned dictionaries are merged into state using state channel reducers
* `messages` field uses `add_messages` reducer (appends new messages)
* Other fields use default replacement semantics unless custom reducer defined
* `jump_to` field is ephemeral (not persisted across invocations)

== Usage Examples ==

'''Example 1: Before Agent Hook - Session Initialization'''

<source lang="python">
from langchain.agents.middleware.types import AgentMiddleware

class SessionMiddleware(AgentMiddleware):
    def before_agent(self, state, runtime):
        """Initialize session tracking."""
        user_id = runtime.context.get("user_id", "anonymous")
        return {
            "session_id": generate_session_id(),
            "user_id": user_id,
            "start_time": datetime.now().isoformat()
        }

    async def abefore_agent(self, state, runtime):
        """Async version loads user preferences from database."""
        user_id = runtime.context.get("user_id")
        preferences = await db.load_user_preferences(user_id)
        return {
            "session_id": generate_session_id(),
            "user_preferences": preferences,
            "start_time": datetime.now().isoformat()
        }
</source>

'''Example 2: Before Model Hook - Context Injection'''

<source lang="python">
from langchain.agents.middleware.types import before_model
from langchain_core.messages import SystemMessage

@before_model
def inject_context(state, runtime):
    """Add system message with current time before each model call."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    context_msg = SystemMessage(content=f"Current time: {current_time}")
    return {"messages": [context_msg]}
</source>

'''Example 3: Wrap Model Call - Retry Logic'''

<source lang="python">
from langchain.agents.middleware.types import wrap_model_call
import time

@wrap_model_call
def retry_with_backoff(request, handler):
    """Retry model calls with exponential backoff."""
    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            return handler(request)
        except Exception as e:
            if attempt == max_retries - 1:
                # Final attempt failed
                raise

            delay = base_delay * (2 ** attempt)
            print(f"Retry {attempt + 1}/{max_retries} after {delay}s: {e}")
            time.sleep(delay)
</source>

'''Example 4: After Model Hook - Conditional Exit'''

<source lang="python">
from langchain.agents.middleware.types import after_model, hook_config
from langchain_core.messages import AIMessage

@after_model(can_jump_to=["end"])
def early_exit_on_marker(state, runtime):
    """Exit agent loop if response contains completion marker."""
    last_msg = state["messages"][-1]

    if isinstance(last_msg, AIMessage) and "[DONE]" in last_msg.content:
        print("Early exit triggered by completion marker")
        return {"jump_to": "end"}

    return None
</source>

'''Example 5: Wrap Tool Call - Timeout and Fallback'''

<source lang="python">
from langchain.agents.middleware.types import wrap_tool_call
from langchain_core.messages import ToolMessage
import asyncio

@wrap_tool_call
async def tool_timeout(request, handler):
    """Enforce timeout on tool execution with fallback."""
    try:
        # Set 10 second timeout
        result = await asyncio.wait_for(handler(request), timeout=10.0)
        return result
    except asyncio.TimeoutError:
        print(f"Tool {request.tool_call['name']} timed out")
        return ToolMessage(
            content="Tool execution timed out",
            tool_call_id=request.tool_call["id"],
            status="error"
        )
</source>

'''Example 6: After Agent Hook - Cleanup and Logging'''

<source lang="python">
from langchain.agents.middleware.types import AgentMiddleware

class CleanupMiddleware(AgentMiddleware):
    def after_agent(self, state, runtime):
        """Log final state and perform cleanup."""
        session_id = state.get("session_id")
        message_count = len(state["messages"])

        # Log session summary
        log_session(session_id, message_count)

        # Cleanup temporary resources
        if "temp_files" in state:
            cleanup_temp_files(state["temp_files"])

        return None
</source>

'''Example 7: Multiple Hooks in Single Middleware'''

<source lang="python">
from langchain.agents.middleware.types import AgentMiddleware, ModelResponse

class MonitoringMiddleware(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.model_call_count = 0
        self.tool_call_count = 0

    def before_agent(self, state, runtime):
        """Reset counters at start."""
        self.model_call_count = 0
        self.tool_call_count = 0
        return None

    def wrap_model_call(self, request, handler):
        """Count model calls and track latency."""
        self.model_call_count += 1
        start = time.time()

        response = handler(request)

        latency = time.time() - start
        metrics.record("model_latency", latency)

        return response

    def wrap_tool_call(self, request, handler):
        """Count tool calls."""
        self.tool_call_count += 1
        return handler(request)

    def after_agent(self, state, runtime):
        """Report final metrics."""
        print(f"Agent execution complete:")
        print(f"  Model calls: {self.model_call_count}")
        print(f"  Tool calls: {self.tool_call_count}")
        return None
</source>

== Related Pages ==

'''Principle:'''
* [[langchain-ai_langchain_Middleware_Lifecycle_Hooks]] - Hook execution semantics

'''Related Implementations:'''
* [[langchain-ai_langchain_AgentMiddleware_base]] - Base class and decorators
* [[langchain-ai_langchain_chain_handlers]] - Handler composition for wrappers
* [[langchain-ai_langchain_state_schema_extension]] - State management

'''Related Principles:'''
* [[langchain-ai_langchain_Middleware_Definition]] - Middleware abstraction
* [[langchain-ai_langchain_Middleware_Composition_Order]] - Multi-middleware orchestration

[[Category:Implementations]]
[[Category:Agent Execution]]
[[Category:Hook Systems]]
