"""
Main entry point for QA model deployment.
"""

from qa_model import predict as _predict


def predict(inputs: dict) -> dict:
    """
    Main entry point for predictions.

    Args:
        inputs: Input dictionary with 'question' and 'context'

    Returns:
        Dictionary with answer and score
    """
    try:
        result = _predict(inputs)
        if "error" in result:
            return {"status": "error", "error": result["error"]}
        return {"status": "success", "output": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    import json
    import sys

    # Read from stdin or use default test input
    if not sys.stdin.isatty():
        input_data = json.loads(sys.stdin.read())
    else:
        input_data = {
            "question": "What is the capital of France?",
            "context": "Paris is the capital and most populous city of France."
        }

    result = predict(input_data)
    print(json.dumps(result, indent=2))
