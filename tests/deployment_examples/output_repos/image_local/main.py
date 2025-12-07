"""
Main entry point for local deployment.
Image processing REST API with Pillow.
"""

import json
import sys


def predict(inputs: dict) -> dict:
    """
    Process images and return results.

    Args:
        inputs: Dictionary with input data
                - image_data: Base64 encoded image (required)
                - width: Target width for resizing (optional)
                - filters: List of filters to apply, e.g., ["grayscale"] (optional)

    Returns:
        Dictionary with results containing processed image data
    """
    try:
        # Import processor module (lazy loading)
        from processor import ImageProcessor

        # Validate required inputs
        if "image_data" not in inputs:
            return {
                "status": "error",
                "error": "Missing required field: image_data"
            }

        # Process the image
        processor = ImageProcessor()
        result = processor.process(
            image_data=inputs["image_data"],
            width=inputs.get("width"),
            filters=inputs.get("filters"),
        )

        return {
            "status": "success",
            "output": result
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
        input_data = {}
        print("Image processor ready. Provide input via stdin.")
        print('Example: echo \'{"image_data": "base64_encoded_image"}\' | python main.py')
    else:
        try:
            input_data = json.loads(sys.stdin.read())
        except json.JSONDecodeError as e:
            print(json.dumps({
                "status": "error",
                "error": f"Invalid JSON input: {str(e)}"
            }, indent=2))
            sys.exit(1)

    result = predict(input_data)
    print(json.dumps(result, indent=2))
