"""
Main entry point for local deployment.
Sentiment analysis using TextBlob.
"""

import json
import sys


def predict(inputs: dict) -> dict:
    """
    Process inputs and return sentiment analysis results.

    Args:
        inputs: Dictionary with input data. Expected keys:
                - 'text': str - Text to analyze

    Returns:
        Dictionary with results including sentiment, polarity, and subjectivity
    """
    try:
        # Import sentiment analyzer (lazy loading)
        from sentiment import SentimentAnalyzer

        # Handle different input formats
        if isinstance(inputs, str):
            text = inputs
        elif "text" in inputs:
            text = inputs["text"]
        else:
            return {"status": "error", "error": "Missing 'text' field in input"}

        # Analyze sentiment
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(text)

        return {"status": "success", "output": result}

    except Exception as e:
        return {"status": "error", "error": str(e)}


# CLI support
if __name__ == "__main__":
    # Read from stdin or use test data
    if sys.stdin.isatty():
        input_data = {"text": "This is a test message."}
    else:
        stdin_content = sys.stdin.read().strip()
        if stdin_content:
            input_data = json.loads(stdin_content)
        else:
            input_data = {"text": "This is a test message."}

    result = predict(input_data)
    print(json.dumps(result, indent=2))
