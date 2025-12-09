"""Main entry point for question answering service."""

import json
import sys
from typing import Dict, Any


class QAModel:
    """Question answering using transformers."""

    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad"):
        self.model_name = model_name
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline("question-answering", model=self.model_name)

    def answer(self, question: str, context: str) -> dict:
        """Answer a question from context."""
        self._load()
        result = self._pipeline(question=question, context=context)
        return {
            "answer": result["answer"],
            "score": round(result["score"], 4),
        }


def predict(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for predictions.

    Args:
        inputs: Input dictionary with 'question' and 'context' keys

    Returns:
        Dictionary with results
    """
    try:
        model = QAModel()

        if "question" in inputs and "context" in inputs:
            result = model.answer(inputs["question"], inputs["context"])
            return {"status": "success", "output": result}

        return {"status": "error", "error": "Provide 'question' and 'context'"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# For CLI usage
if __name__ == "__main__":
    input_data = json.loads(sys.stdin.read()) if not sys.stdin.isatty() else {}
    result = predict(input_data)
    print(json.dumps(result))
