"""
Main entry point for local deployment.
Production ML text classification with batching support.
"""

import json
import sys


def predict(inputs: dict) -> dict:
    """
    Process inputs and return classification results.

    Args:
        inputs: Dictionary with input data
                Expected keys: 'text' (str) or 'texts' (list of str)

    Returns:
        Dictionary with results
    """
    try:
        # Import classifier module (lazy loading for better startup time)
        from classifier import predict as classify

        # Process inputs
        result = classify(inputs)

        # Wrap result in standard format
        return {"status": "success", "output": result}

    except Exception as e:
        return {"status": "error", "error": str(e)}


# CLI support
if __name__ == "__main__":
    # Read from stdin or use empty dict
    if sys.stdin.isatty():
        input_data = {"text": "This is a test input"}
    else:
        input_data = json.loads(sys.stdin.read())

    result = predict(input_data)
    print(json.dumps(result, indent=2))
