# Text Embeddings - ORIGINAL INPUT
#
# Uses sentence-transformers with GPU support.
# NO deployment files - just core logic.

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


def predict(inputs):
    """Main prediction function for GPU embeddings."""
    embedder = TextEmbedder()
    
    if isinstance(inputs, str):
        return {"embedding": embedder.embed(inputs), "dimension": 384}
    
    if "text" in inputs:
        emb = embedder.embed(inputs["text"])
        return {"embedding": emb, "dimension": len(emb)}
    
    return {"error": "Invalid input"}


if __name__ == "__main__":
    print("Embedder ready (requires GPU)")

