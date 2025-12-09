"""
FastAPI application for Docker deployment.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict

app = FastAPI(title="Image Processing API", version="1.0.0")


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
def predict_endpoint(request: Dict[str, Any]):
    """
    Main prediction endpoint.

    IMPORTANT: Accepts raw JSON input directly (not wrapped in "data").
    Example: {"image_data": "base64..."} NOT {"data": {"image_data": "base64..."}}
    """
    try:
        from processor import predict
        result = predict(request)
        return PredictResponse(status="success", output=result)
    except Exception as e:
        return PredictResponse(status="error", error=str(e))
