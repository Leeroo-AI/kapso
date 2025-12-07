"""
BentoML service for BentoCloud deployment.
"""

import bentoml
from typing import Any, Dict, List, Union


@bentoml.service(
    name="text-classifier-service",
    resources={
        "cpu": "2",
        "memory": "4Gi",
    },
    traffic={
        "timeout": 300,
    },
)
class TextClassifierService:
    """
    BentoML service wrapping the text classifier.
    Deployed to BentoCloud for managed scaling.
    """

    def __init__(self):
        """Initialize the service."""
        from main import predict as _predict
        self._predict = _predict

    @bentoml.api
    def predict(self, inputs: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """
        Main prediction API.

        Args:
            inputs: Input dictionary with "text" or "texts", or a string

        Returns:
            Result dictionary with classification
        """
        try:
            result = self._predict(inputs)
            return {"status": "success", "output": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=32,
        max_latency_ms=1000,
    )
    def predict_batch(self, inputs: List[str]) -> List[Dict[str, Any]]:
        """
        Batched prediction for throughput optimization.
        BentoCloud automatically batches requests.

        Args:
            inputs: List of text strings to classify

        Returns:
            List of classification results
        """
        try:
            results = self._predict(inputs)
            return results
        except Exception as e:
            return [{"status": "error", "error": str(e)}] * len(inputs)

    @bentoml.api
    def health(self) -> Dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy"}
