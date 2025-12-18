{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangGraph|https://langchain-ai.github.io/langgraph/]]
* [[source::Doc|LangChain Agents|https://docs.langchain.com/oss/python/langchain/overview]]
|-
! Domains
| [[domain::LLM]], [[domain::Agents]], [[domain::Graph_Computing]], [[domain::Execution]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Concrete tool for executing compiled agent graphs through invoke, stream, and async variants, provided by LangGraph's execution runtime.

=== Description ===

`CompiledStateGraph.invoke`, `stream`, `ainvoke`, and `astream` are the execution methods for running agent graphs. These methods:
* Execute the graph starting from the initial state
* Run nodes in sequence following edges and conditional routing
* Handle tool calling and response parsing
* Support checkpointing for persistence and resumption
* Enable streaming of intermediate results

The execution implements the agent loop: model call → tool execution → model call → ... → termination.

=== Usage ===

Use these methods when:
* Running an agent to completion (`invoke`)
* Streaming intermediate results for real-time UI (`stream`)
* Executing agents in async contexts (`ainvoke`, `astream`)
* Resuming from interrupted execution

Method selection:
* **invoke:** Blocking, returns final state
* **stream:** Yields events during execution
* **ainvoke:** Async blocking
* **astream:** Async streaming

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain]
* '''File:''' libs/langchain_v1/langchain/agents/factory.py
* '''Lines:''' L1116-1196 (model_node/amodel_node functions)
* '''LangGraph:''' langgraph.graph.state.CompiledStateGraph

=== Signature ===
<syntaxhighlight lang="python">
class CompiledStateGraph:
    """Compiled graph ready for execution."""

    def invoke(
        self,
        input: dict | None,
        config: RunnableConfig | None = None,
        *,
        stream_mode: Literal["values", "updates", "debug"] = "values",
        output_keys: str | Sequence[str] | None = None,
        interrupt_before: Sequence[str] | None = None,
        interrupt_after: Sequence[str] | None = None,
        debug: bool | None = None,
    ) -> dict:
        """Execute graph to completion.

        Args:
            input: Initial state (must contain "messages" key).
            config: Runtime configuration with thread_id for persistence.
            stream_mode: How to aggregate streamed results.
            output_keys: Which state keys to return.
            interrupt_before: Nodes to pause before (overrides compile-time).
            interrupt_after: Nodes to pause after (overrides compile-time).
            debug: Enable debug logging.

        Returns:
            Final state dictionary with messages and optional structured_response.
        """

    def stream(
        self,
        input: dict | None,
        config: RunnableConfig | None = None,
        *,
        stream_mode: Literal["values", "updates", "debug"] = "values",
        output_keys: str | Sequence[str] | None = None,
        interrupt_before: Sequence[str] | None = None,
        interrupt_after: Sequence[str] | None = None,
        debug: bool | None = None,
    ) -> Iterator[dict]:
        """Stream execution events.

        Yields:
            State updates or values depending on stream_mode.
        """

    async def ainvoke(
        self,
        input: dict | None,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict:
        """Async execute graph to completion."""

    async def astream(
        self,
        input: dict | None,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> AsyncIterator[dict]:
        """Async stream execution events."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
# The agent returned is a CompiledStateGraph

# For direct LangGraph access:
from langgraph.graph.state import CompiledStateGraph
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| input || dict | None || Yes* || Initial state with "messages" key (*None to resume from checkpoint)
|-
| config || RunnableConfig | None || No || Config with "configurable.thread_id" for persistence
|-
| stream_mode || "values" | "updates" | "debug" || No || How to yield results (default: "values")
|-
| output_keys || str | Sequence[str] | None || No || Filter returned state keys
|-
| interrupt_before || Sequence[str] | None || No || Nodes to pause before
|-
| interrupt_after || Sequence[str] | None || No || Nodes to pause after
|-
| debug || bool | None || No || Enable debug logging
|}

=== Outputs (invoke/ainvoke) ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| messages || list[AnyMessage] || Full message history including user, assistant, and tool messages
|-
| structured_response || SchemaT | None || Parsed structured output if response_format was configured
|}

=== Outputs (stream/astream) ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| event || dict || State update or full state depending on stream_mode
|}

== Usage Examples ==

=== Basic Invocation ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

agent = create_agent(
    model=init_chat_model("gpt-4o"),
    tools=[...],
)

# Invoke with initial messages
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "What's the weather in NYC?"}
    ]
})

# Access results
print(result["messages"][-1].content)  # Final assistant message
</syntaxhighlight>

=== Streaming for Real-Time UI ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

agent = create_agent(
    model=init_chat_model("gpt-4o"),
    tools=[...],
)

# Stream execution events
for event in agent.stream(
    {"messages": [{"role": "user", "content": "Search for Python tutorials"}]},
    stream_mode="updates"  # Only yield changes
):
    # Process each event as it occurs
    if "messages" in event:
        for msg in event["messages"]:
            print(f"[{msg.type}] {msg.content[:100]}")
</syntaxhighlight>

=== Async Execution ===
<syntaxhighlight lang="python">
import asyncio
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model


async def run_agent():
    agent = create_agent(
        model=init_chat_model("gpt-4o"),
        tools=[...],
    )

    # Async invoke
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Hello!"}]
    })
    return result


# Run
result = asyncio.run(run_agent())
</syntaxhighlight>

=== Async Streaming ===
<syntaxhighlight lang="python">
import asyncio
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model


async def stream_agent():
    agent = create_agent(
        model=init_chat_model("gpt-4o"),
        tools=[...],
    )

    async for event in agent.astream({
        "messages": [{"role": "user", "content": "Explain quantum computing"}]
    }):
        print(event)


asyncio.run(stream_agent())
</syntaxhighlight>

=== Persistent Conversation ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

agent = create_agent(
    model=init_chat_model("gpt-4o"),
    tools=[...],
    checkpointer="memory",
)

# Thread ID for conversation persistence
config = {"configurable": {"thread_id": "user-session-123"}}

# First message
result = agent.invoke(
    {"messages": [{"role": "user", "content": "My name is Alice"}]},
    config=config
)

# Follow-up (agent remembers previous context)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    config=config
)
# Agent correctly responds "Alice"
</syntaxhighlight>

=== Resume After Interruption ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

agent = create_agent(
    model=init_chat_model("gpt-4o"),
    tools=[dangerous_tool],
    interrupt_before=["tools"],
    checkpointer="memory",
)

config = {"configurable": {"thread_id": "approval-123"}}

# First invoke - pauses before tools
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Delete all files"}]},
    config=config
)

# User reviews pending tool call...
# result["messages"][-1].tool_calls shows what will be executed

# Resume execution (pass None as input to continue)
result = agent.invoke(None, config=config)
</syntaxhighlight>

=== With Structured Output ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from pydantic import BaseModel


class Answer(BaseModel):
    text: str
    confidence: float


agent = create_agent(
    model=init_chat_model("gpt-4o"),
    tools=[],
    response_format=ToolStrategy(Answer),
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is 2+2?"}]
})

# Access typed structured response
answer: Answer = result["structured_response"]
print(f"Answer: {answer.text} (confidence: {answer.confidence})")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:langchain-ai_langchain_Agent_Loop_Execution]]

=== Requires Environment ===
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
