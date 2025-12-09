"""BentoML service for text classification on BentoCloud."""

import bentoml
from typing import Any, Dict, List, Union


@bentoml.service(
    name="text-classifier",
    resources={"cpu": "2", "memory": "4Gi"},
    traffic={"timeout": 300},
)
class TextClassifierService:
    """BentoML service for text classification with batching support."""

    def __init__(self):
        from main import predict as _predict
        self._predict = _predict

    @bentoml.api
    def predict(self, inputs: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """
        Predict endpoint for text classification.

        Args:
            inputs: Input data. Can be:
                    - {"text": "single text"} for single prediction
                    - {"texts": ["text1", "text2"]} for batch predictions

        Returns:
            Dictionary with status and output
        """
        try:
            # Handle string input (convert to dict)
            if isinstance(inputs, str):
                inputs = {"text": inputs}

            result = self._predict(inputs)
            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @bentoml.api
    def health(self) -> Dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy"}
