"""
Main entry point for question answering model.
"""

from qa_model import QAModel


def predict(inputs: dict) -> dict:
    """
    Main entry point for predictions.

    Args:
        inputs: Input dictionary with 'question' and 'context'

    Returns:
        Dictionary with answer and score
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
    import json
    import sys

    input_data = json.loads(sys.stdin.read()) if not sys.stdin.isatty() else {}
    result = predict(input_data)
    print(json.dumps(result))
