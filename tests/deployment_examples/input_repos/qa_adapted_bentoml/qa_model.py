# Question Answering - ORIGINAL INPUT
#
# Uses transformers pipeline for QA.
# NO deployment files - just core logic.


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


def predict(inputs):
    """Main prediction function for QA."""
    model = QAModel()
    
    if "question" in inputs and "context" in inputs:
        return model.answer(inputs["question"], inputs["context"])
    
    return {"error": "Provide 'question' and 'context'"}


if __name__ == "__main__":
    print("QA model ready")

