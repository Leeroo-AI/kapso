"""
Main entry point for local deployment.
Text embeddings API using sentence-transformers with GPU support.
"""

import json
import sys


def predict(inputs: dict) -> dict:
    """
    Process inputs and return text embeddings.

    Args:
        inputs: Dictionary with input data. Expected keys:
                - 'text': String to embed
                - Or string input will be converted to {'text': input}

    Returns:
        Dictionary with results containing:
        - 'status': 'success' or 'error'
        - 'output': embedding and dimension info
    """
    try:
        # Import embedder module (lazy loading)
        from embedder import TextEmbedder

        # Initialize embedder
        embedder = TextEmbedder()

        # Handle string input
        if isinstance(inputs, str):
            emb = embedder.embed(inputs)
            return {
                "status": "success",
                "output": {
                    "embedding": emb,
                    "dimension": len(emb)
                }
            }

        # Handle dictionary input with 'text' key
        if "text" in inputs:
            emb = embedder.embed(inputs["text"])
            return {
                "status": "success",
                "output": {
                    "embedding": emb,
                    "dimension": len(emb)
                }
            }

        # Handle similarity calculation if both texts provided
        if "text1" in inputs and "text2" in inputs:
            similarity = embedder.similarity(inputs["text1"], inputs["text2"])
            return {
                "status": "success",
                "output": {
                    "similarity": similarity
                }
            }

        return {
            "status": "error",
            "error": "Invalid input. Expected 'text' key or 'text1' and 'text2' keys."
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# CLI support
if __name__ == "__main__":
    # Read from stdin or use empty dict
    if sys.stdin.isatty():
        input_data = {"text": "Hello world"}
    else:
        input_data = json.loads(sys.stdin.read())

    result = predict(input_data)
    print(json.dumps(result, indent=2))
