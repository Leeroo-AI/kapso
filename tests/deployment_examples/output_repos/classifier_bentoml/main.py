"""
Main entry point for text classification.
This module provides the predict() function for all deployment targets.
"""

import json
import sys
from classifier import TextClassifier


def predict(inputs):
    """
    Main entry point for predictions.

    Args:
        inputs: Input dictionary, string, or list
                - dict with "text": single classification
                - dict with "texts": batch classification
                - str: single classification
                - list: batch classification

    Returns:
        Dictionary or list with classification results
    """
    classifier = TextClassifier()

    # Handle batch input (list)
    if isinstance(inputs, list):
        texts = [inp.get("text", inp) if isinstance(inp, dict) else inp for inp in inputs]
        return classifier.classify_batch(texts)

    # Handle single input (string)
    if isinstance(inputs, str):
        return classifier.classify(inputs)

    # Handle dictionary input
    if isinstance(inputs, dict):
        if "text" in inputs:
            return classifier.classify(inputs["text"])
        if "texts" in inputs:
            return classifier.classify_batch(inputs["texts"])

    return {"error": "Invalid input. Provide 'text' or 'texts'."}


# For CLI usage
if __name__ == "__main__":
    try:
        if not sys.stdin.isatty():
            stdin_content = sys.stdin.read().strip()
            if stdin_content:
                input_data = json.loads(stdin_content)
            else:
                input_data = {"text": "This is a great product!"}
        else:
            # Test examples
            input_data = {"text": "This is a great product!"}

        result = predict(input_data)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}, indent=2))
        sys.exit(1)
