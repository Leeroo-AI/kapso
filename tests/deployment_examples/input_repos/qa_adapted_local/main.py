"""
Main entry point for local deployment.
Question answering using transformers pipeline.
"""

def predict(inputs: dict) -> dict:
    """
    Process inputs and return question answering results.

    Args:
        inputs: Dictionary with 'question' and 'context' keys

    Returns:
        Dictionary with 'answer' and 'score' or error message
    """
    try:
        # Import from existing qa_model module
        from qa_model import predict as qa_predict

        # Call the existing predict function
        result = qa_predict(inputs)

        # Ensure consistent response format
        if "error" in result:
            return {"status": "error", "error": result["error"]}

        return {"status": "success", "output": result}

    except Exception as e:
        return {"status": "error", "error": str(e)}


# CLI support
if __name__ == "__main__":
    import json
    import sys

    # Read from stdin or use empty dict
    if sys.stdin.isatty():
        input_data = {}
    else:
        input_data = json.loads(sys.stdin.read())

    result = predict(input_data)
    print(json.dumps(result, indent=2))
