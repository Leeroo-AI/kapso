"""
Main entry point for local deployment.

Conversational AI chatbot using LangGraph with memory.
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
import os


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


# Build and compile the graph (lazy initialization will happen in predict)
_graph = None


def _get_graph():
    """Get or initialize the graph."""
    global _graph
    if _graph is None:
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("chatbot", chatbot_node)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        _graph = graph_builder.compile()
    return _graph


def predict(inputs: dict) -> dict:
    """
    Process inputs and return results.

    Args:
        inputs: Dictionary with input data. Supports:
            - {"messages": [...]} - List of message objects
            - {"text": "..."} - Simple text input
            - {"content": "..."} - Content input
            - {"test": True} - Test mode (returns success)

    Returns:
        Dictionary with results: {"status": "success", "output": ...}
    """
    try:
        # Handle test mode
        if inputs.get("test"):
            return {
                "status": "success",
                "output": {
                    "message": "Chatbot is ready",
                    "type": "test_response"
                }
            }

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
                # Default: convert entire dict to string
                messages = [{"role": "user", "content": str(inputs)}]
        else:
            messages = [{"role": "user", "content": str(inputs)}]

        # Get the compiled graph
        graph = _get_graph()

        # Run the graph
        result = graph.invoke({"messages": messages})

        # Convert result to JSON-serializable format
        output_messages = []
        for msg in result.get("messages", []):
            # Convert LangChain message objects to dicts
            if hasattr(msg, "content"):
                output_messages.append({
                    "role": getattr(msg, "type", "unknown"),
                    "content": msg.content
                })
            else:
                output_messages.append(msg)

        return {
            "status": "success",
            "output": {
                "messages": output_messages
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


# CLI support
if __name__ == "__main__":
    import json
    import sys

    # Read from stdin or use empty dict
    if sys.stdin.isatty():
        input_data = {"test": True}
    else:
        input_data = json.loads(sys.stdin.read())

    result = predict(input_data)
    print(json.dumps(result, indent=2))
