"""
Text Embeddings API - Main Entry Point
Uses sentence-transformers with GPU support for generating text embeddings.
"""

import numpy as np
from typing import Union


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


def predict(inputs: Union[dict, str]) -> dict:
    """
    Main entry point for text embeddings.

    Args:
        inputs: Either a string or dict with 'text' key

    Returns:
        Dictionary with embedding and dimension
    """
    try:
        embedder = TextEmbedder()

        if isinstance(inputs, str):
            emb = embedder.embed(inputs)
            return {"status": "success", "embedding": emb, "dimension": len(emb)}

        if isinstance(inputs, dict) and "text" in inputs:
            emb = embedder.embed(inputs["text"])
            return {"status": "success", "embedding": emb, "dimension": len(emb)}

        return {"status": "error", "error": "Invalid input. Provide 'text' key or string."}

    except Exception as e:
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    import json
    import sys

    if not sys.stdin.isatty():
        input_data = json.loads(sys.stdin.read())
    else:
        input_data = {"text": "Hello, world!"}

    result = predict(input_data)
    print(json.dumps(result))
