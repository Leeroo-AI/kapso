"""
Main entry point for sentiment analysis.
"""

import json
import sys
from sentiment import predict as sentiment_predict


def predict(inputs: dict) -> dict:
    """
    Main entry point for predictions/processing.

    Args:
        inputs: Input dictionary with data to process

    Returns:
        Dictionary with results
    """
    try:
        result = sentiment_predict(inputs)
        return {"status": "success", "output": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# For CLI usage
if __name__ == "__main__":
    if not sys.stdin.isatty():
        input_data = json.loads(sys.stdin.read())
    else:
        # Default test input
        input_data = {"text": "I love this!"}

    result = predict(input_data)
    print(json.dumps(result, indent=2))
