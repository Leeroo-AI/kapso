# Text Classification - ORIGINAL INPUT
#
# Uses scikit-learn for text classification.
# Designed for BentoML deployment with batching support.
# NO deployment files - just core logic.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np


class TextClassifier:
    """
    Text classifier using TF-IDF + Logistic Regression.
    
    Designed for production ML serving with:
    - Batching support
    - Model versioning
    - Fast inference
    """
    
    # Sample training data (in production, load from file/database)
    TRAIN_TEXTS = [
        "This product is amazing and works great",
        "Terrible quality, waste of money",
        "Excellent customer service",
        "Very disappointed with the purchase",
        "Would recommend to everyone",
        "Never buying from here again",
        "Best purchase I ever made",
        "Broke after one day of use",
        "Five stars, absolutely love it",
        "Zero stars if I could",
    ]
    TRAIN_LABELS = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative
    
    def __init__(self):
        self.model = None
        self._trained = False
    
    def _ensure_trained(self):
        """Lazy training on first use."""
        if not self._trained:
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000)),
                ('clf', LogisticRegression(max_iter=1000)),
            ])
            self.model.fit(self.TRAIN_TEXTS, self.TRAIN_LABELS)
            self._trained = True
    
    def classify(self, text: str) -> dict:
        """Classify a single text."""
        self._ensure_trained()
        
        prediction = self.model.predict([text])[0]
        proba = self.model.predict_proba([text])[0]
        
        return {
            "text": text,
            "label": "positive" if prediction == 1 else "negative",
            "confidence": round(float(max(proba)), 4),
        }
    
    def classify_batch(self, texts: list) -> list:
        """Classify multiple texts efficiently (for batching)."""
        self._ensure_trained()
        
        predictions = self.model.predict(texts)
        probas = self.model.predict_proba(texts)
        
        results = []
        for text, pred, proba in zip(texts, predictions, probas):
            results.append({
                "text": text,
                "label": "positive" if pred == 1 else "negative",
                "confidence": round(float(max(proba)), 4),
            })
        return results


def predict(inputs):
    """
    Main prediction function.
    
    Supports both single and batch predictions.
    """
    classifier = TextClassifier()
    
    # Handle batch input
    if isinstance(inputs, list):
        texts = [inp.get("text", inp) if isinstance(inp, dict) else inp for inp in inputs]
        return classifier.classify_batch(texts)
    
    # Handle single input
    if isinstance(inputs, str):
        return classifier.classify(inputs)
    
    if isinstance(inputs, dict):
        if "text" in inputs:
            return classifier.classify(inputs["text"])
        if "texts" in inputs:
            return classifier.classify_batch(inputs["texts"])
    
    return {"error": "Invalid input. Provide 'text' or 'texts'."}


if __name__ == "__main__":
    # Test the classifier
    result = predict({"text": "This is a great product!"})
    print(f"Single: {result}")
    
    results = predict({"texts": ["I love it", "Terrible experience"]})
    print(f"Batch: {results}")

