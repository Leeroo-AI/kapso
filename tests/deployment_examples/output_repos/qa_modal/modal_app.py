"""
Modal application for serverless QA model deployment.
"""

import modal

# Define the Modal app with a unique name
app = modal.App("qa-model")

# Define the container image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.100.0",  # Required for web endpoints
        "transformers>=4.30.0",
        "torch>=2.0.0",
    )
    # Mount local Python files into the image
    .add_local_file("main.py", "/root/main.py")
    .add_local_file("qa_model.py", "/root/qa_model.py")
)


@app.function(
    image=image,
    gpu="T4",  # T4 GPU as specified in requirements
    memory=16384,  # 16Gi = 16384 MB
    timeout=300,
)
def predict(inputs: dict) -> dict:
    """
    Main prediction function deployed to Modal.

    Args:
        inputs: Input dictionary with 'question' and 'context'

    Returns:
        Result dictionary with answer and score
    """
    import sys
    sys.path.insert(0, "/root")
    from main import predict as _predict
    return _predict(inputs)


# Web endpoint for HTTP access
@app.function(
    image=image,
    gpu="T4",  # Same GPU config as predict
    memory=16384,
    timeout=300,
)
@modal.fastapi_endpoint(method="POST")
def web_predict(inputs: dict) -> dict:
    """
    Web endpoint for HTTP POST requests.

    Usage:
        curl -X POST https://[your-modal-url].modal.run \
             -H "Content-Type: application/json" \
             -d '{"question": "What is AI?", "context": "Artificial Intelligence is..."}'
    """
    import sys
    sys.path.insert(0, "/root")
    from main import predict as _predict
    return _predict(inputs)


# Local testing entrypoint
@app.local_entrypoint()
def main():
    """Test the function locally before deploying."""
    print("Testing Modal QA deployment...")
    test_input = {
        "question": "What is the capital of France?",
        "context": "Paris is the capital and most populous city of France."
    }
    result = predict.remote(test_input)
    print(f"Result: {result}")
