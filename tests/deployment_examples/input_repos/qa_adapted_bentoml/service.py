"""BentoML service for BentoCloud deployment."""

import bentoml
from typing import Any, Dict, Union


@bentoml.service(
    name="qa-service",
    resources={"cpu": "2", "memory": "4Gi"},
    traffic={"timeout": 300},
)
class QAService:
    def __init__(self):
        from main import predict as _predict
        self._predict = _predict

    @bentoml.api
    def predict(self, inputs: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        try:
            result = self._predict(inputs)
            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @bentoml.api
    def health(self) -> Dict[str, str]:
        return {"status": "healthy"}
