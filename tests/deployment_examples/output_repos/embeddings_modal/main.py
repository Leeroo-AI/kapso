"""
Main entry point for text embeddings API.
"""

import numpy as np


class TextEmbedder:
    """Text embedder using sentence-transformers with GPU."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(self.model_name, device=device)

    def embed(self, text: str) -> list:
        """Generate embedding for text."""
        self._load()
        return self._model.encode(text, convert_to_numpy=True).tolist()

    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity."""
        emb1 = np.array(self.embed(text1))
        emb2 = np.array(self.embed(text2))
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


def predict(inputs: dict) -> dict:
    """
    Main entry point for predictions/processing.

    Args:
        inputs: Input dictionary with data to process

    Returns:
        Dictionary with results
    """
    try:
        embedder = TextEmbedder()

        # Handle string input
        if isinstance(inputs, str):
            emb = embedder.embed(inputs)
            return {"status": "success", "embedding": emb, "dimension": len(emb)}

        # Handle dict with "text" key
        if "text" in inputs:
            emb = embedder.embed(inputs["text"])
            return {"status": "success", "embedding": emb, "dimension": len(emb)}

        # Handle dict with "text1" and "text2" for similarity
        if "text1" in inputs and "text2" in inputs:
            similarity = embedder.similarity(inputs["text1"], inputs["text2"])
            return {"status": "success", "similarity": similarity}

        return {"status": "error", "error": "Invalid input. Provide 'text' or 'text1'/'text2'"}

    except Exception as e:
        return {"status": "error", "error": str(e)}


# For CLI usage
if __name__ == "__main__":
    import json
    import sys

    input_data = json.loads(sys.stdin.read()) if not sys.stdin.isatty() else {"text": "Hello world"}
    result = predict(input_data)
    print(json.dumps(result))
