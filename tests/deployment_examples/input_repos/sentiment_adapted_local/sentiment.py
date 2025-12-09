# Sentiment Analysis Module - ORIGINAL INPUT
#
# Uses TextBlob for sentiment analysis.
# NO deployment files - just core logic.

from textblob import TextBlob


class SentimentAnalyzer:
    """Sentiment analyzer using TextBlob."""
    
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
    
    def analyze(self, text: str) -> dict:
        """Analyze sentiment of text."""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > self.threshold:
            sentiment = "positive"
        elif polarity < -self.threshold:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "text": text,
            "polarity": round(polarity, 4),
            "subjectivity": round(subjectivity, 4),
            "sentiment": sentiment,
        }


def predict(inputs):
    """Main prediction function."""
    analyzer = SentimentAnalyzer()
    
    if isinstance(inputs, str):
        return analyzer.analyze(inputs)
    
    if "text" in inputs:
        return analyzer.analyze(inputs["text"])
    
    return {"error": "Invalid input"}


if __name__ == "__main__":
    print(predict({"text": "I love this!"}))

