"""Main entry point for LangGraph agent."""
from agent import graph


def predict(inputs):
    """
    Main prediction function.

    Args:
        inputs: Input dictionary or string

    Returns:
        Dictionary with status and output
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
    import json
    import sys

    # Test with sample input
    if not sys.stdin.isatty():
        input_data = json.loads(sys.stdin.read())
    else:
        input_data = {"text": "Hello!"}

    result = predict(input_data)
    print(json.dumps(result, default=str))
