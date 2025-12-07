# Text Embeddings API - Modal Deployment

GPU-accelerated text embeddings API using sentence-transformers, deployed on Modal.

## Features

- Text embeddings using `all-MiniLM-L6-v2` model
- GPU acceleration (T4)
- Serverless scaling
- HTTP API endpoint

## Local Testing

Install dependencies:
```bash
pip install -r requirements.txt
```

Test the predict function:
```bash
echo '{"text": "Hello world"}' | python main.py
```

Test similarity:
```bash
echo '{"text1": "Hello world", "text2": "Hi there"}' | python main.py
```

## Modal Deployment

### Setup

1. Install Modal:
```bash
pip install modal
```

2. Authenticate (first time only):
```bash
modal token new
```

### Deploy

Deploy to Modal:
```bash
modal deploy modal_app.py
```

### Test Deployed Function

Test via Modal CLI:
```bash
modal run modal_app.py
```

### Call via HTTP

After deployment, you'll get a URL. Use it like this:

```bash
curl -X POST "https://[your-username]--text-embeddings-api-web-predict.modal.run" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world"}'
```

### Call via Python SDK

```python
import modal

# Look up the deployed function
predict = modal.Function.lookup("text-embeddings-api", "predict")

# Call it remotely
result = predict.remote({"text": "Hello world"})
print(result)
```

## API

### Input Format

**Single text embedding:**
```json
{
  "text": "Your text here"
}
```

**Similarity between two texts:**
```json
{
  "text1": "First text",
  "text2": "Second text"
}
```

### Output Format

**Embedding response:**
```json
{
  "status": "success",
  "embedding": [0.1, 0.2, ...],
  "dimension": 384
}
```

**Similarity response:**
```json
{
  "status": "success",
  "similarity": 0.85
}
```

**Error response:**
```json
{
  "status": "error",
  "error": "Error message"
}
```

## Resources

- GPU: T4
- Memory: 16GB RAM
- Timeout: 5 minutes
