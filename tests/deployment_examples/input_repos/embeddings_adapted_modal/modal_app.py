"""
Modal application for serverless text embeddings deployment.
Deploys sentence-transformers model with GPU support.
"""

import modal

# Define the Modal app with a unique name
app = modal.App("embeddings-api")

# Define the container image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.100.0",
        "torch>=2.0.0",
        "sentence-transformers>=2.2.0",
        "numpy>=1.21.0",
    )
    .add_local_file("main.py", "/root/main.py")
)


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    memory=16384,
)
def predict(inputs: dict) -> dict:
    """Main prediction function deployed to Modal."""
    import sys
    sys.path.insert(0, "/root")
    from main import predict as _predict
    return _predict(inputs)


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    memory=16384,
)
@modal.fastapi_endpoint(method="POST")
def web_predict(inputs: dict) -> dict:
    """Web endpoint for HTTP POST requests."""
    import sys
    sys.path.insert(0, "/root")
    from main import predict as _predict
    return _predict(inputs)


@app.local_entrypoint()
def main():
    """Test the function locally before deploying."""
    print("Testing Modal deployment...")
    test_input = {"text": "Hello, world! This is a test."}
    result = predict.remote(test_input)
    print(f"Result: {result}")
