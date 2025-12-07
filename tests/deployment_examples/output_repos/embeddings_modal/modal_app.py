"""
Modal application for serverless text embeddings API.
"""

import modal

# Define the Modal app with a unique name
app = modal.App("text-embeddings-api")

# Define the container image with dependencies
# IMPORTANT: Include fastapi for web endpoints
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.100.0",  # Required for web endpoints
        "torch>=2.0.0",
        "sentence-transformers>=2.2.0",
        "numpy>=1.21.0",
    )
    # Mount local Python files into the image
    .add_local_file("main.py", "/root/main.py")
)


@app.function(
    image=image,
    gpu="T4",  # T4 GPU for inference
    memory=16384,  # 16GB RAM
    timeout=300,
)
def predict(inputs: dict) -> dict:
    """
    Main prediction function deployed to Modal.

    Args:
        inputs: Input dictionary

    Returns:
        Result dictionary
    """
    import sys
    sys.path.insert(0, "/root")
    from main import predict as _predict
    return _predict(inputs)


# Web endpoint for HTTP access
# IMPORTANT: Use @modal.fastapi_endpoint (not @modal.web_endpoint which is deprecated)
@app.function(
    image=image,
    gpu="T4",  # Same GPU config as predict
    memory=16384,  # 16GB RAM
    timeout=300,
)
@modal.fastapi_endpoint(method="POST")
def web_predict(inputs: dict) -> dict:
    """
    Web endpoint for HTTP POST requests.

    Usage:
        curl -X POST https://[your-modal-url].modal.run \
             -H "Content-Type: application/json" \
             -d '{"text": "Hello world"}'
    """
    import sys
    sys.path.insert(0, "/root")
    from main import predict as _predict
    return _predict(inputs)


# Local testing entrypoint
@app.local_entrypoint()
def main():
    """Test the function locally before deploying."""
    print("Testing Modal deployment...")
    test_input = {"text": "This is a test sentence for embeddings."}
    result = predict.remote(test_input)
    print(f"Result: {result}")
