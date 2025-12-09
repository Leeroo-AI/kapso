"""
Main entry point for local deployment.
Image processing REST API with Pillow.
"""

import json
import sys


def predict(inputs: dict) -> dict:
    """
    Process inputs and return results.

    Args:
        inputs: Dictionary with input data
            - image_data: Base64-encoded image data
            - width: Optional width for resizing
            - filters: Optional list of filters to apply

    Returns:
        Dictionary with results including processed image
    """
    try:
        # Import processor module (lazy loading)
        from processor import predict as processor_predict

        # Validate inputs
        if not inputs:
            return {"status": "error", "error": "No input provided"}

        # Call the processor
        result = processor_predict(inputs)

        # Check if there was an error from processor
        if "error" in result:
            return {"status": "error", "error": result["error"]}

        # Return successful result
        return {"status": "success", "output": result}

    except Exception as e:
        return {"status": "error", "error": str(e)}


# CLI support
if __name__ == "__main__":
    # Read from stdin or use empty dict
    if sys.stdin.isatty():
        input_data = {"test": True}
    else:
        input_data = json.loads(sys.stdin.read())

    result = predict(input_data)
    print(json.dumps(result, indent=2))
