# Docker Deployment Instructions

Docker deployment runs the solution in an isolated container.
Best for reproducibility, isolation, and HTTP-based APIs.

## DEPLOY COMMAND

```bash
docker build -t solution . && docker run -d --name solution-container -p 8000:8000 solution
```

Run this command to build and start the Docker container. The `-d` runs in detached mode, `--name` gives a consistent container name.
If it fails, debug and fix the error.

## RUN INTERFACE
- type: http
- endpoint: http://localhost:8000
- path: /predict
- container_name: solution-container
- image_name: solution

After successful deployment, output this JSON (update endpoint if different):
```
<run_interface>{"type": "http", "endpoint": "http://localhost:8000", "path": "/predict", "container_name": "solution-container", "image_name": "solution"}</run_interface>
```

## CRITICAL: YOU MUST BUILD AND TEST THE CONTAINER

**Do NOT just create files. You MUST build and verify the Docker image works.**

After creating the Dockerfile:
1. Run: `docker build -t solution .`
2. Verify the build succeeds without errors
3. Run: `docker run -p 8000:8000 solution` to start the container
4. Test with: `curl http://localhost:8000/health`
5. If there are errors, fix them and rebuild

**If deployment fails, debug the error and fix it. Do not give up.**

## Required Structure

```
solution/
├── main.py           # Entry point with predict() function
├── app.py            # FastAPI application (for HTTP interface)
├── requirements.txt  # Dependencies
├── Dockerfile        # Container definition
└── .dockerignore     # Files to exclude from image
```

## Dockerfile

Create a `Dockerfile` with this content:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for HTTP interface
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## FastAPI Application (app.py)

If using HTTP interface, create `app.py`:

```python
"""
FastAPI application for Docker deployment.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict

app = FastAPI(title="Solution API", version="1.0.0")


class PredictResponse(BaseModel):
    """Output schema for predictions."""
    status: str
    output: Any = None
    error: str = None


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: Dict[str, Any]):
    """
    Main prediction endpoint.
    
    IMPORTANT: Accepts raw JSON input directly (not wrapped in "data").
    Example: {"text": "hello"} NOT {"data": {"text": "hello"}}
    """
    try:
        from main import predict as _predict
        result = _predict(request)
        return PredictResponse(status="success", output=result)
    except Exception as e:
        return PredictResponse(status="error", error=str(e))
```

## .dockerignore

Create `.dockerignore` to exclude unnecessary files:

```
__pycache__
*.pyc
*.pyo
.git
.gitignore
.env
*.md
tests/
.pytest_cache/
.mypy_cache/
```

## Requirements

Add FastAPI and uvicorn to `requirements.txt`:

```
fastapi>=0.100.0
uvicorn>=0.23.0
# ... your other dependencies
```

## Testing Docker Locally

```bash
# Build the image
docker build -t solution .

# Run the container
docker run -p 8000:8000 solution

# Test health endpoint
curl http://localhost:8000/health

# Test prediction endpoint (send input directly, not wrapped in "data")
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "hello"}'
```

## Notes

- Use multi-stage builds for smaller images if needed
- Pin base image versions for reproducibility
- Don't include secrets in the image - use env vars at runtime

