# LangGraph Platform Deployment Instructions

LangGraph Platform deploys stateful AI agents with built-in memory, streaming, and scaling.
Best for complex multi-step agents, conversational AI, and agentic workflows.

## DEPLOY COMMAND

```bash
langgraph deploy
```

Run this command to deploy to LangGraph Platform. Capture the deployment URL from the output.
If deployment fails, debug and fix the error.

## RUN INTERFACE
- type: langgraph
- assistant_id: agent
- deployment_url: from deployment output

After successful deployment, output this JSON (update deployment_url):
```
<run_interface>{"type": "langgraph", "assistant_id": "agent", "deployment_url": "https://your-deployment-url"}</run_interface>
<endpoint_url>https://your-deployment-url</endpoint_url>
```

## CRITICAL: YOU MUST ACTUALLY DEPLOY

**Do NOT just create files. You MUST run `langgraph deploy` and verify it succeeds.**

After creating the LangGraph files:
1. Run: `langgraph deploy`
2. Wait for deployment to complete
3. Capture the deployment URL
4. Report the deployment status

**If deployment fails, debug the error and fix it. Do not give up.**

## Environment Variables Required

- `LANGSMITH_API_KEY` - Your LangSmith API key (required)
- `ANTHROPIC_API_KEY` - If using Claude (required for the agent)

## Required Structure

```
solution/
├── agent.py           # LangGraph agent with exported graph
├── main.py            # Entry point with predict()
├── langgraph.json     # LangGraph configuration
├── requirements.txt   # Dependencies
└── .env               # Environment variables
```

## Main Entry Point (main.py)

```python
"""Main entry point for LangGraph agent."""
from agent import graph


def predict(inputs):
    """Main prediction function."""
    if isinstance(inputs, str):
        messages = [{"role": "user", "content": inputs}]
    elif isinstance(inputs, dict):
        if "messages" in inputs:
            messages = inputs["messages"]
        elif "text" in inputs:
            messages = [{"role": "user", "content": inputs["text"]}]
        else:
            messages = [{"role": "user", "content": str(inputs)}]
    else:
        messages = [{"role": "user", "content": str(inputs)}]
    
    result = graph.invoke({"messages": messages})
    return {"status": "success", "output": result}


if __name__ == "__main__":
    result = predict({"text": "Hello!"})
    print(result)
```

## Agent Definition (agent.py)

```python
"""LangGraph agent for deployment."""
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def agent_node(state: AgentState) -> dict:
    llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_edge(START, "agent")
graph_builder.add_edge("agent", END)

# Export the compiled graph
graph = graph_builder.compile()
```

## Configuration (langgraph.json)

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./agent.py:graph"
  },
  "env": ".env"
}
```

## Requirements

```
langgraph>=0.2.0
langchain-anthropic>=0.2.0
langchain-core>=0.3.0
langgraph-sdk>=0.1.0
```

## Deployment Commands

```bash
# Install CLI
pip install langgraph-cli

# Set API key
export LANGSMITH_API_KEY="your-api-key"

# Deploy
langgraph deploy
```

## Calling the Deployed Agent

```python
from langgraph_sdk import get_client

client = get_client(url="YOUR_DEPLOYMENT_URL")

# Create thread
thread = await client.threads.create()

# Send message
result = await client.runs.wait(
    thread["thread_id"],
    "agent",
    input={"messages": [{"role": "user", "content": "Hello!"}]}
)
```

## Notes

- The `graph` variable must be exported for LangGraph Platform
- Use `add_messages` annotation for automatic message history
- Threads persist conversation state automatically

