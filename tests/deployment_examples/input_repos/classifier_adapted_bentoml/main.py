"""Main entry point for text classification."""

from classifier import TextClassifier


def predict(inputs: dict) -> dict:
    """
    Main entry point for predictions.

    Args:
        inputs: Input dictionary with data to process.
                Supports:
                - {"text": "single text"} -> single prediction
                - {"texts": ["text1", "text2"]} -> batch predictions

    Returns:
        Dictionary with results
    """
    try:
        classifier = TextClassifier()

        # Handle batch input
        if "texts" in inputs and isinstance(inputs["texts"], list):
            results = classifier.classify_batch(inputs["texts"])
            return {"status": "success", "output": results}

        # Handle single input
        if "text" in inputs:
            result = classifier.classify(inputs["text"])
            return {"status": "success", "output": result}

        return {"status": "error", "error": "Invalid input. Provide 'text' or 'texts'."}

    except Exception as e:
        return {"status": "error", "error": str(e)}


# For CLI usage
if __name__ == "__main__":
    import json
    import sys

    if not sys.stdin.isatty():
        input_data = json.loads(sys.stdin.read())
    else:
        # Test with sample data
        input_data = {"text": "This is a great product!"}

    result = predict(input_data)
    print(json.dumps(result, indent=2))
