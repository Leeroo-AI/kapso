# Chatbot Agent - ORIGINAL INPUT
#
# Simple chatbot using LangGraph.
# NO deployment files - just core agent logic.

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic


class AgentState(TypedDict):
    """
    Agent state with message history.
    
    The add_messages annotation enables automatic message accumulation
    across conversation turns.
    """
    messages: Annotated[list, add_messages]


def chatbot_node(state: AgentState) -> dict:
    """
    Main chatbot node that processes messages.
    
    Args:
        state: Current agent state with messages
        
    Returns:
        Updated state with new AI message
    """
    # Initialize Claude (uses ANTHROPIC_API_KEY from environment)
    llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    
    # Process messages and generate response
    response = llm.invoke(state["messages"])
    
    return {"messages": [response]}


# Build the state graph
graph_builder = StateGraph(AgentState)

# Add the chatbot node
graph_builder.add_node("chatbot", chatbot_node)

# Define edges: START -> chatbot -> END
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph for deployment
# This is what LangGraph Platform will import
graph = graph_builder.compile()


def predict(inputs):
    """
    Main prediction function for local testing.
    
    Args:
        inputs: Dict with "messages" key or string
        
    Returns:
        Agent response
    """
    # Handle different input formats
    if isinstance(inputs, str):
        messages = [{"role": "user", "content": inputs}]
    elif isinstance(inputs, dict):
        if "messages" in inputs:
            messages = inputs["messages"]
        elif "text" in inputs:
            messages = [{"role": "user", "content": inputs["text"]}]
        elif "content" in inputs:
            messages = [{"role": "user", "content": inputs["content"]}]
        else:
            messages = [{"role": "user", "content": str(inputs)}]
    else:
        messages = [{"role": "user", "content": str(inputs)}]
    
    # Run the graph
    result = graph.invoke({"messages": messages})
    
    return {
        "status": "success",
        "output": result,
    }


if __name__ == "__main__":
    # Test locally
    result = predict({"text": "Hello! What's your name?"})
    print(result)


