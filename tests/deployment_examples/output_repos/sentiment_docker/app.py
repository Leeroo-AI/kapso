"""
FastAPI application for Docker deployment.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, Optional

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")


class PredictRequest(BaseModel):
    """Input schema for predictions."""
    text: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class PredictResponse(BaseModel):
    """Output schema for predictions."""
    status: str
    output: Optional[Any] = None
    error: Optional[str] = None


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "service": "Sentiment Analysis API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Main prediction endpoint.

    Accepts either:
    - {"text": "some text to analyze"}
    - {"data": {"text": "some text to analyze"}}
    """
    try:
        from main import predict as _predict

        # Handle different input formats
        if request.text:
            result = _predict({"text": request.text})
        elif request.data:
            result = _predict(request.data)
        else:
            return PredictResponse(
                status="error",
                error="Please provide either 'text' or 'data' field"
            )

        return PredictResponse(status="success", output=result)
    except Exception as e:
        return PredictResponse(status="error", error=str(e))
