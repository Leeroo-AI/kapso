"""
FastAPI application for Docker deployment.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")


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
def predict(request: dict):
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
