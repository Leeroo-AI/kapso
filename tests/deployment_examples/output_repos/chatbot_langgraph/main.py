"""
Main entry point for LangGraph chatbot agent.

This module provides a standardized predict() interface for local testing
and deployment compatibility.
"""
import json
import sys
from agent import graph


def predict(inputs):
    """
    Main prediction function for the chatbot.

    Args:
        inputs: Dict with "messages", "text", or "content" key, or a string

    Returns:
        Dict with status and agent response
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

    try:
        # Run the graph
        result = graph.invoke({"messages": messages})

        return {
            "status": "success",
            "output": result,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    # Support both stdin and direct testing
    if not sys.stdin.isatty():
        input_data = json.loads(sys.stdin.read())
    else:
        input_data = {"text": "Hello! What can you help me with?"}

    result = predict(input_data)
    print(json.dumps(result, indent=2, default=str))
