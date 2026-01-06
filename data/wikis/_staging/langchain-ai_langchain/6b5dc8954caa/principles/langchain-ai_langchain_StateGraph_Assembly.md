{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|LangGraph|https://langchain-ai.github.io/langgraph/]]
* [[source::Doc|State Machines|https://en.wikipedia.org/wiki/Finite-state_machine]]
* [[source::Doc|LangChain Agents|https://docs.langchain.com/oss/python/langchain/overview]]
|-
! Domains
| [[domain::LLM]], [[domain::Agents]], [[domain::Graph_Computing]], [[domain::State_Machines]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Design pattern for constructing directed acyclic graphs (DAGs) that define agent execution flow with nodes, edges, and conditional routing.

=== Description ===

StateGraph Assembly is the process of building a computational graph that orchestrates agent execution. Unlike simple function composition, graph-based execution enables:
* **Conditional branching:** Different paths based on model output
* **Cycles:** The agent loop (model → tools → model → ...)
* **Persistence:** Checkpoint and resume execution at any node
* **Interruption:** Human-in-the-loop at defined points
* **Parallelism:** Execute independent nodes concurrently

This principle transforms the classic "agent loop" from imperative code into a declarative graph structure that can be inspected, modified, and executed with guarantees.

=== Usage ===

Use StateGraph Assembly when:
* Building agents with complex control flow
* Implementing human-in-the-loop patterns
* Creating persistent, resumable workflows
* Designing multi-agent systems
* Requiring visibility into execution structure

The graph abstraction makes agent behavior:
* Deterministic (same graph = same behavior)
* Debuggable (inspect graph structure)
* Testable (mock individual nodes)
* Composable (graphs as nodes in larger graphs)

== Theoretical Basis ==

StateGraph Assembly implements **Stateful Graph Computation** for agent orchestration.

'''1. Graph Components'''

<syntaxhighlight lang="python">
# Pseudo-code for graph components
class StateGraph:
    state_schema: type  # TypedDict defining graph state
    nodes: dict[str, Callable]  # Node name -> function
    edges: list[tuple[str, str]]  # Static edges
    conditional_edges: list[tuple[str, Callable, dict]]  # Conditional routing

    def add_node(self, name: str, func: Callable):
        """Add a node that transforms state."""
        self.nodes[name] = func

    def add_edge(self, source: str, target: str):
        """Add unconditional edge."""
        self.edges.append((source, target))

    def add_conditional_edges(
        self,
        source: str,
        router: Callable[[State], str],
        destinations: dict[str, str]
    ):
        """Add conditional routing from source."""
        self.conditional_edges.append((source, router, destinations))
</syntaxhighlight>

'''2. Agent Loop as Graph'''

The classic agent loop translates to:

<syntaxhighlight lang="text">
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   START ──► model ──►(has_tools?)──► tools ──┐              │
│                │                              │              │
│                │ no                           │              │
│                ▼                              │              │
│               END ◄──────────────────────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
</syntaxhighlight>

<syntaxhighlight lang="python">
# Pseudo-code for agent graph
def build_agent_graph():
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("model", call_model)
    graph.add_node("tools", execute_tools)

    # Edges
    graph.add_edge(START, "model")
    graph.add_conditional_edges(
        "model",
        lambda state: "tools" if state.last_message.has_tool_calls else END,
        {"tools": "tools", END: END}
    )
    graph.add_edge("tools", "model")  # Loop back

    return graph.compile()
</syntaxhighlight>

'''3. State Flow'''

Each node receives state and returns updates:

<syntaxhighlight lang="python">
# Node function signature
def node_function(state: AgentState) -> dict[str, Any]:
    """Process state and return updates.

    Args:
        state: Current graph state (immutable view)

    Returns:
        Dictionary of state updates to merge
    """
    # Process
    result = process(state)
    # Return updates (not full state)
    return {"messages": [result_message], "some_field": new_value}

# State updates are merged, not replaced
# new_state = {**old_state, **updates}
</syntaxhighlight>

'''4. Conditional Routing'''

Routers determine next node based on state:

<syntaxhighlight lang="python">
def route_after_model(state: AgentState) -> str:
    """Decide next node after model execution."""
    last_message = state["messages"][-1]

    # Check for tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Check for structured output (ends agent)
    if state.get("structured_response"):
        return END

    # Check for jump directive from middleware
    if state.get("jump_to"):
        return state["jump_to"]

    return END
</syntaxhighlight>

'''5. Compilation and Execution'''

Compilation validates and optimizes the graph:

<syntaxhighlight lang="python">
# Pseudo-code for compilation
def compile(graph, checkpointer=None):
    # Validate graph structure
    validate_no_orphan_nodes(graph)
    validate_reachability(graph)

    # Create executable
    compiled = CompiledStateGraph(
        graph=graph,
        checkpointer=checkpointer,
    )

    # Execution methods
    compiled.invoke(input)  # Sync execution
    compiled.ainvoke(input)  # Async execution
    compiled.stream(input)  # Stream events
    compiled.astream(input)  # Async stream

    return compiled
</syntaxhighlight>

'''6. Persistence with Checkpointing'''

Checkpointers enable persistence and resumption:

<syntaxhighlight lang="python">
# Pseudo-code for checkpointing
class Checkpointer:
    def put(self, config, checkpoint):
        """Save checkpoint at current node."""

    def get(self, config):
        """Load checkpoint for resumption."""

# During execution
def execute_with_checkpointing(graph, input, config, checkpointer):
    # Check for existing checkpoint
    checkpoint = checkpointer.get(config)
    if checkpoint:
        state = checkpoint.state
        next_node = checkpoint.next_node
    else:
        state = input
        next_node = START

    # Execute graph
    while next_node != END:
        # Save checkpoint before each node
        checkpointer.put(config, Checkpoint(state, next_node))

        # Execute node
        updates = graph.nodes[next_node](state)
        state = merge(state, updates)

        # Route to next node
        next_node = route(graph, state, next_node)

    return state
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:langchain-ai_langchain_create_agent_graph_building]]

=== Used By Workflows ===
* Agent_Creation_Workflow (Step 5)
