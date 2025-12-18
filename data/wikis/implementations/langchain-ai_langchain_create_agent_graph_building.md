{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangGraph|https://langchain-ai.github.io/langgraph/]]
* [[source::Doc|LangChain Agents|https://docs.langchain.com/oss/python/langchain/overview]]
|-
! Domains
| [[domain::LLM]], [[domain::Agents]], [[domain::Graph_Computing]], [[domain::State_Machines]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Concrete tool for assembling LangGraph StateGraph structures that define agent execution flow, provided by LangChain's agent factory.

=== Description ===

The `create_agent` function internally builds a LangGraph `StateGraph` that orchestrates agent execution. This includes:
* Adding model and tool nodes to the graph
* Wiring middleware nodes at appropriate lifecycle points
* Setting up conditional edges for tool calling decisions
* Configuring structured output handling
* Compiling to an executable `CompiledStateGraph`

The graph structure implements the classic agent loop: model → tools → model → ... → end.

=== Usage ===

Use StateGraph assembly (via `create_agent`) when:
* Building tool-calling agents
* Creating custom agent architectures
* Implementing complex multi-step workflows
* Adding middleware to agent execution

The internal graph structure is abstracted by `create_agent`; direct StateGraph manipulation is typically not needed.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain]
* '''File:''' libs/langchain_v1/langchain/agents/factory.py
* '''Lines:''' L861-1483

=== Signature ===
<syntaxhighlight lang="python">
def create_agent(
    model: BaseChatModel | _ConfigurableModel,
    tools: Sequence[BaseTool | Callable | dict] = (),
    *,
    middleware: Sequence[AgentMiddleware | type[AgentMiddleware]] | None = None,
    state_schema: type[AgentState[SchemaT]] | None = None,
    response_format: ResponseFormat[SchemaT] | type[SchemaT] | None = None,
    prompt: SystemMessage | str | Callable | None = None,
    tool_choice: Any | None = None,
    checkpointer: BaseCheckpointSaver | Literal["memory"] | None = None,
    store: BaseStore | None = None,
    interrupt_before: Sequence[str] | None = None,
    interrupt_after: Sequence[str] | None = None,
    debug: bool = False,
    name: str = "Agent",
    **model_settings: Any,
) -> CompiledStateGraph:
    """Create an agent with the given model, tools, and configuration.

    Args:
        model: The chat model to use (can be configurable).
        tools: Tools available to the agent.
        middleware: Middleware to intercept agent lifecycle.
        state_schema: Custom state schema (extends AgentState).
        response_format: Structured output configuration.
        prompt: System prompt (string, SystemMessage, or callable).
        tool_choice: Tool selection constraint.
        checkpointer: Persistence for graph state.
        store: Cross-thread storage.
        interrupt_before: Nodes to interrupt before.
        interrupt_after: Nodes to interrupt after.
        debug: Enable verbose logging.
        name: Name for the compiled graph.
        **model_settings: Additional model kwargs.

    Returns:
        CompiledStateGraph ready for invoke/stream execution.
    """
</syntaxhighlight>

=== Internal Graph Structure ===
<syntaxhighlight lang="python">
# Pseudo-code for internal graph assembly
def _build_agent_graph(model, tools, middleware, response_format, ...):
    # 1. Create StateGraph with merged state schema
    merged_schema = merge_state_schemas([AgentState] + [m.state_schema for m in middleware])
    graph = StateGraph(merged_schema, input=InputSchema, output=OutputSchema)

    # 2. Add middleware "before" nodes
    for m in middleware:
        if m.has_before_agent:
            graph.add_node(f"{m.name}_before_agent", m.before_agent)
        if m.has_before_model:
            graph.add_node(f"{m.name}_before_model", m.before_model)

    # 3. Add core nodes
    graph.add_node("model", model_node)  # Calls model with wrapped handlers
    graph.add_node("tools", ToolNode(tools))  # Executes tool calls

    # 4. Add middleware "after" nodes
    for m in middleware:
        if m.has_after_model:
            graph.add_node(f"{m.name}_after_model", m.after_model)

    # 5. Wire edges with conditional routing
    graph.add_edge(START, first_before_agent_node or "model")
    graph.add_conditional_edges("model", route_after_model)  # -> tools or end
    graph.add_conditional_edges("tools", route_after_tools)  # -> model or end

    # 6. Compile
    return graph.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
    )
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
# Or for direct StateGraph access:
from langgraph.graph.state import StateGraph
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || BaseChatModel | _ConfigurableModel || Yes || Chat model for agent reasoning
|-
| tools || Sequence[BaseTool | Callable | dict] || No || Tools available to the agent
|-
| middleware || Sequence[AgentMiddleware] | None || No || Middleware for lifecycle interception
|-
| state_schema || type[AgentState] | None || No || Custom state schema
|-
| response_format || ResponseFormat | type | None || No || Structured output configuration
|-
| prompt || SystemMessage | str | Callable | None || No || System prompt
|-
| tool_choice || Any | None || No || Tool selection constraint
|-
| checkpointer || BaseCheckpointSaver | "memory" | None || No || State persistence
|-
| store || BaseStore | None || No || Cross-thread storage
|-
| interrupt_before || Sequence[str] | None || No || Nodes to interrupt before
|-
| interrupt_after || Sequence[str] | None || No || Nodes to interrupt after
|-
| debug || bool || No || Enable verbose logging
|-
| name || str || No || Graph name (default: "Agent")
|-
| **model_settings || Any || No || Additional model kwargs
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return || CompiledStateGraph || Executable agent graph with invoke/stream/ainvoke/astream methods
|}

== Usage Examples ==

=== Basic Agent Creation ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"


agent = create_agent(
    model=init_chat_model("gpt-4o"),
    tools=[search],
    prompt="You are a helpful research assistant.",
)

result = agent.invoke({"messages": [{"role": "user", "content": "Search for Python tutorials"}]})
</syntaxhighlight>

=== Agent with Middleware and Structured Output ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from pydantic import BaseModel


class ResponseSchema(BaseModel):
    answer: str
    confidence: float


class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        print(f"Calling model with {len(state['messages'])} messages")
        return None


agent = create_agent(
    model=init_chat_model("gpt-4o"),
    tools=[],
    middleware=[LoggingMiddleware()],
    response_format=ToolStrategy(ResponseSchema),
    prompt="Answer questions concisely.",
)
</syntaxhighlight>

=== Agent with Checkpointing (Persistence) ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

# Create agent with memory persistence
agent = create_agent(
    model=init_chat_model("gpt-4o"),
    tools=[...],
    checkpointer="memory",  # Or MemorySaver() or database-backed saver
)

# Invoke with thread ID for persistence
config = {"configurable": {"thread_id": "user-123"}}
result = agent.invoke({"messages": [...]}, config=config)

# Continue conversation in same thread
result = agent.invoke({"messages": [...]}, config=config)
</syntaxhighlight>

=== Agent with Human-in-the-Loop ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

# Interrupt before tool execution for approval
agent = create_agent(
    model=init_chat_model("gpt-4o"),
    tools=[dangerous_tool],
    interrupt_before=["tools"],  # Pause before tool execution
    checkpointer="memory",  # Required for interruption
)

# First invoke - will pause before tools
result = agent.invoke({"messages": [...]}, config={"configurable": {"thread_id": "1"}})

# User reviews and approves - continue execution
result = agent.invoke(None, config={"configurable": {"thread_id": "1"}})
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:langchain-ai_langchain_StateGraph_Assembly]]

=== Requires Environment ===
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
