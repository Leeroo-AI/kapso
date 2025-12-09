"""
Main entry point for sentiment analysis.
"""

from sentiment import predict as sentiment_predict


def predict(inputs: dict) -> dict:
    """
    Main entry point for predictions.

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
    import json
    import sys

    input_data = json.loads(sys.stdin.read()) if not sys.stdin.isatty() else {"text": "This is a test!"}
    result = predict(input_data)
    print(json.dumps(result))
