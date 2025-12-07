"""
Main entry point for local deployment.
Sentiment analysis API using TextBlob.
"""

def predict(inputs: dict) -> dict:
    """
    Process inputs and return sentiment analysis results.

    Args:
        inputs: Dictionary with input data. Expected format:
                {"text": "text to analyze"}
                or list of texts: {"texts": ["text1", "text2"]}

    Returns:
        Dictionary with sentiment analysis results
    """
    # Import sentiment analyzer (lazy loading)
    from sentiment import SentimentAnalyzer

    try:
        analyzer = SentimentAnalyzer()

        # Handle single text input
        if "text" in inputs:
            result = analyzer.analyze(inputs["text"])
            return {"status": "success", "output": result}

        # Handle batch text input
        elif "texts" in inputs:
            results = [analyzer.analyze(text) for text in inputs["texts"]]
            return {"status": "success", "output": results}

        # No valid input provided
        else:
            return {
                "status": "error",
                "error": "Invalid input. Expected 'text' or 'texts' field."
            }

    except Exception as e:
        return {"status": "error", "error": str(e)}


# CLI support
if __name__ == "__main__":
    import json
    import sys

    # Check if running with stdin input
    if len(sys.argv) > 1:
        # Use command line argument
        input_data = json.loads(sys.argv[1])
    elif not sys.stdin.isatty():
        # Data piped via stdin
        input_data = json.loads(sys.stdin.read())
    else:
        # No input - use example
        input_data = {"text": "I love this product!"}
        print("No input provided. Using example:", input_data)

    result = predict(input_data)
    print(json.dumps(result, indent=2))
