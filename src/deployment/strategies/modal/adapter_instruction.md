# Modal Deployment Instructions

Modal deployment runs the solution on Modal.com's serverless infrastructure.
Best for GPU workloads, auto-scaling, and serverless execution.

## DEPLOY COMMAND

```bash
modal deploy modal_app.py
```

Run this command to deploy to Modal. Capture the endpoint URL from the output (e.g., `https://username--app-name-web-predict.modal.run`).
If deployment fails, debug and fix the error.

## RUN INTERFACE
- type: modal
- app_name: derived from path
- callable: predict

After successful deployment, output this JSON (update app_name and endpoint):
```
<run_interface>{"type": "modal", "app_name": "your-app-name", "callable": "predict"}</run_interface>
<endpoint_url>https://your-username--app-name-web-predict.modal.run</endpoint_url>
```

## CRITICAL: YOU MUST ACTUALLY DEPLOY

**Do NOT just create files. You MUST run `modal deploy modal_app.py` and verify it succeeds.**

After creating the Modal app:
1. Run: `modal deploy modal_app.py`
2. Wait for deployment to complete
3. Capture the endpoint URL from the output
4. Test the endpoint with curl to verify it works
5. Report the endpoint URL

**If deployment fails, debug the error and fix it. Do not give up.**

## Required Structure

```
solution/
├── main.py           # Core logic with predict() function
├── modal_app.py      # Modal application definition
├── requirements.txt  # Dependencies
└── ...               # Other modules
```

## Modal Application (modal_app.py)

Create `modal_app.py` with this structure:

```python
"""
Modal application for serverless deployment.
"""

import modal

# Define the Modal app with a unique name
app = modal.App("solution-name")

# Define the container image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.100.0",  # Required for web endpoints
        # Add your other dependencies here
    )
    .add_local_file("main.py", "/root/main.py")
)


@app.function(
    image=image,
    # gpu="T4",  # Uncomment if GPU needed: "T4", "A10G", "A100"
    timeout=300,
)
def predict(inputs: dict) -> dict:
    """Main prediction function deployed to Modal."""
    import sys
    sys.path.insert(0, "/root")
    from main import predict as _predict
    return _predict(inputs)


@app.function(image=image, timeout=300)
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
    result = predict.remote({"test": True})
    print(f"Result: {result}")
```

## GPU Configuration

If your solution needs GPU, update the `@app.function` decorator:

```python
@app.function(
    image=image,
    gpu="T4",  # Options: "T4", "L4", "A10G", "A100", "H100"
    timeout=600,
    memory=16384,  # 16GB RAM
)
def predict(inputs: dict) -> dict:
    ...
```

## Image with Local Files

Always mount your Python files into the image:

```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.100.0",
        "torch>=2.0.0",
        "numpy>=1.21.0",
    )
    .add_local_file("main.py", "/root/main.py")
    .add_local_file("model.py", "/root/model.py")
)
```

## Secrets Management

For API keys and secrets, use Modal secrets:

```python
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("my-secret")],
)
def predict(inputs: dict) -> dict:
    import os
    api_key = os.environ["API_KEY"]
    ...
```

## Testing Modal Locally

```bash
# Install Modal
pip install modal

# Authenticate (first time only)
modal token new

# Test locally
modal run modal_app.py

# Deploy
modal deploy modal_app.py
```

## Notes

- Modal automatically handles scaling and cold starts
- ALWAYS use `@modal.fastapi_endpoint` for web endpoints
- ALWAYS mount local files with `.add_local_file()`
- ALWAYS add path to sys.path before importing
- Monitor usage in the Modal dashboard

