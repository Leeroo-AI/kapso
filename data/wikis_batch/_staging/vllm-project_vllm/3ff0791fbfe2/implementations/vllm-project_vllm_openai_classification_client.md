# OpenAI Classification API Client Example

**Source:** `examples/pooling/classify/openai_classification_client.py`
**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)
**Lines:** 53

## Overview

This example demonstrates how to use vLLM's classification API endpoint, which is compatible with OpenAI-style API patterns. It shows how to send text prompts to a classification model and receive predicted class labels, making it easy to integrate text classification capabilities into applications.

## Implementation Pattern

### Architecture Design

The client uses a simple HTTP-based request/response pattern:

**Client Side:**
- Constructs JSON payload with model name and input texts
- Posts to `/classify` endpoint
- Receives classification results with predicted labels

**Server Side (vLLM):**
- Serves classification model via API endpoint
- Processes batch of texts in parallel
- Returns structured classification responses

### Use Cases

**Text Classification Tasks:**
- Sentiment analysis (positive/negative/neutral)
- Topic categorization
- Intent detection for chatbots
- Content moderation
- Language detection
- Spam filtering

## Technical Implementation

### 1. HTTP Request Function

```python
def post_http_request(payload: dict, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    response = requests.post(api_url, headers=headers, json=payload)
    return response
```

**Purpose:**
Sends a POST request to the classification endpoint with proper headers.

**Headers:**
- `User-Agent`: Identifies the client application
- `Content-Type: application/json`: Implicit from `json=payload` parameter

**Returns:**
A `requests.Response` object containing the classification results.

### 2. Command-Line Arguments

```python
def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--host", type=str, default="localhost")
    parse.add_argument("--port", type=int, default=8000)
    parse.add_argument("--model", type=str, default="jason9693/Qwen2.5-1.5B-apeach")
    return parse.parse_args()
```

**Configuration Options:**

**--host:**
- Server hostname or IP address
- Default: `localhost` (local server)
- Example: `--host vllm-server.example.com`

**--port:**
- Server port number
- Default: `8000` (standard vLLM port)
- Example: `--port 8080`

**--model:**
- Model identifier for classification
- Default: `jason9693/Qwen2.5-1.5B-apeach` (example classification model)
- Must match model loaded by vLLM server

### 3. Main Classification Logic

```python
def main(args):
    host = args.host
    port = args.port
    model_name = args.model

    api_url = f"http://{host}:{port}/classify"
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    payload = {
        "model": model_name,
        "input": prompts,
    }

    classify_response = post_http_request(payload=payload, api_url=api_url)
    pprint.pprint(classify_response.json())
```

**Workflow:**

1. **Construct URL:** Builds endpoint URL from host and port
2. **Prepare Prompts:** Creates list of texts to classify
3. **Build Payload:** Packages model name and inputs into request
4. **Send Request:** Posts to classification endpoint
5. **Display Results:** Pretty-prints the JSON response

## Request Format

### Payload Structure

```json
{
  "model": "jason9693/Qwen2.5-1.5B-apeach",
  "input": [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is"
  ]
}
```

**Fields:**

**model (required):**
- String identifier for the classification model
- Must match the model served by vLLM
- Example: `"jason9693/Qwen2.5-1.5B-apeach"`

**input (required):**
- List of strings to classify
- Can be a single string or array of strings
- Batch processing for efficiency

**Optional Fields:**
While the example doesn't use them, the API may support:
- `temperature`: Controls randomness for probabilistic classifiers
- `top_k`: Returns top K class predictions
- `logprobs`: Returns class probabilities

## Response Format

### Expected Response Structure

```json
{
  "id": "classify-request-id",
  "object": "classification",
  "created": 1234567890,
  "model": "jason9693/Qwen2.5-1.5B-apeach",
  "data": [
    {
      "object": "classification",
      "index": 0,
      "label": "greeting",
      "score": 0.95
    },
    {
      "object": "classification",
      "index": 1,
      "label": "politics",
      "score": 0.89
    },
    {
      "object": "classification",
      "index": 2,
      "label": "geography",
      "score": 0.97
    },
    {
      "object": "classification",
      "index": 3,
      "label": "technology",
      "score": 0.92
    }
  ]
}
```

**Response Fields:**

**id:**
Unique identifier for this classification request.

**object:**
Response type, typically `"classification"`.

**created:**
Unix timestamp of when the request was processed.

**model:**
Echo of the model name used.

**data:**
Array of classification results, one per input.

**Per-Result Fields:**
- `object`: Result type (`"classification"`)
- `index`: Position in the input array
- `label`: Predicted class label
- `score`: Confidence score (0-1)

## Usage Requirements

### Starting the vLLM Server

Before running the client, start a vLLM server with a classification model:

```bash
vllm serve jason9693/Qwen2.5-1.5B-apeach \
  --trust-remote-code \
  --port 8000
```

**Server Configuration:**

**--trust-remote-code:**
Required for models with custom code (like many classification models).

**--port:**
Port to bind the API server (default: 8000).

**Additional Options:**
- `--host`: Bind address (default: `0.0.0.0`)
- `--model-name`: Override the model identifier used in API requests

### Running the Client

```bash
# Basic usage (default settings)
python openai_classification_client.py

# Custom server
python openai_classification_client.py \
  --host vllm-server.example.com \
  --port 8080

# Different model
python openai_classification_client.py \
  --model my-custom-classifier
```

### Dependencies

```python
import argparse
import pprint
import requests
```

**Installation:**
```bash
pip install requests
```

The client has minimal dependencies and works with Python 3.7+.

## Classification Models

### Supported Model Types

**Sequence Classification Models:**
Models with a classification head that outputs class probabilities.

**Example Models:**
- `jason9693/Qwen2.5-1.5B-apeach`: Sentiment analysis
- `distilbert-base-uncased-finetuned-sst-2-english`: Binary sentiment
- `facebook/bart-large-mnli`: Natural language inference
- Custom fine-tuned classifiers

### Model Requirements

**Architecture:**
Must be a model with a classification head (not generative).

**Output:**
Should produce class labels or probabilities.

**HuggingFace Compatibility:**
Works with models that follow HuggingFace transformers conventions.

## Production Patterns

### Error Handling

```python
def post_http_request(payload: dict, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response
    except requests.exceptions.Timeout:
        print(f"Request timed out to {api_url}")
        raise
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
        print(f"Response: {e.response.text}")
        raise
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        raise
```

### Batch Processing

```python
def classify_texts_in_batches(texts, batch_size=32):
    """Process large text lists in batches."""
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = {
            "model": model_name,
            "input": batch,
        }

        response = post_http_request(payload, api_url)
        results.extend(response.json()["data"])

    return results
```

### Async Processing

```python
import asyncio
import aiohttp

async def classify_async(session, payload, api_url):
    """Async classification request."""
    async with session.post(api_url, json=payload) as response:
        return await response.json()

async def classify_many_async(text_batches, api_url):
    """Process multiple batches concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [
            classify_async(
                session,
                {"model": model_name, "input": batch},
                api_url
            )
            for batch in text_batches
        ]
        return await asyncio.gather(*tasks)
```

### Retries with Exponential Backoff

```python
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def create_session_with_retries():
    """Create a requests session with retry logic."""
    session = requests.Session()

    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )

    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session

# Usage
session = create_session_with_retries()
response = session.post(api_url, headers=headers, json=payload)
```

## Integration Examples

### Flask Web Application

```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

VLLM_API_URL = "http://localhost:8000/classify"
MODEL_NAME = "jason9693/Qwen2.5-1.5B-apeach"

@app.route('/classify', methods=['POST'])
def classify():
    """Classify text via vLLM backend."""
    data = request.json
    texts = data.get('texts', [])

    if not texts:
        return jsonify({"error": "No texts provided"}), 400

    payload = {
        "model": MODEL_NAME,
        "input": texts,
    }

    try:
        response = requests.post(VLLM_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
```

### Streamlit Dashboard

```python
import streamlit as st
import requests

st.title("Text Classification Demo")

text_input = st.text_area("Enter text to classify:", height=150)

if st.button("Classify"):
    if text_input:
        payload = {
            "model": "jason9693/Qwen2.5-1.5B-apeach",
            "input": [text_input],
        }

        with st.spinner("Classifying..."):
            response = requests.post(
                "http://localhost:8000/classify",
                json=payload
            )

        if response.ok:
            result = response.json()["data"][0]
            st.success(f"Label: {result['label']}")
            st.metric("Confidence", f"{result['score']:.2%}")
        else:
            st.error("Classification failed")
    else:
        st.warning("Please enter some text")
```

### Command-Line Tool

```python
#!/usr/bin/env python3
"""CLI tool for text classification."""

import sys
import json
import requests
from typing import List

def classify_from_stdin(api_url: str, model: str):
    """Read texts from stdin and classify."""
    texts = [line.strip() for line in sys.stdin if line.strip()]

    if not texts:
        print("No input provided", file=sys.stderr)
        sys.exit(1)

    payload = {"model": model, "input": texts}
    response = requests.post(api_url, json=payload)

    if response.ok:
        results = response.json()["data"]
        for result in results:
            print(json.dumps(result))
    else:
        print(f"Error: {response.text}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    classify_from_stdin(
        "http://localhost:8000/classify",
        "jason9693/Qwen2.5-1.5B-apeach"
    )
```

**Usage:**
```bash
# Classify lines from file
cat texts.txt | python classify_cli.py

# Classify command output
echo "This is great!" | python classify_cli.py
```

## Performance Optimization

### Batching Strategy

**Optimal Batch Sizes:**
- Small model: 32-64 texts per batch
- Medium model: 16-32 texts per batch
- Large model: 8-16 texts per batch

**Trade-offs:**
- Larger batches: Better throughput, higher latency
- Smaller batches: Lower latency, less throughput

### Connection Pooling

```python
import requests

# Create persistent session
session = requests.Session()

# Reuse session for multiple requests
for batch in batches:
    response = session.post(api_url, json=payload)
    process_response(response)
```

Benefits:
- TCP connection reuse
- Reduced handshake overhead
- Better performance for high-frequency requests

## Troubleshooting

### Common Issues

**Connection Refused:**
```
requests.exceptions.ConnectionError: Connection refused
```

**Solution:**
- Verify vLLM server is running: `curl http://localhost:8000/health`
- Check firewall settings
- Confirm port number matches

**Model Not Found:**
```
{"error": "Model not found"}
```

**Solution:**
- Ensure model name matches server configuration
- Use `vllm serve --help` to check model naming

**Timeout Errors:**
```
requests.exceptions.Timeout
```

**Solution:**
- Increase timeout: `requests.post(..., timeout=60)`
- Check server load with `nvidia-smi`
- Reduce batch size

**Invalid Response:**
```
json.decoder.JSONDecodeError
```

**Solution:**
- Check response status: `response.raise_for_status()`
- Verify endpoint URL (should be `/classify`)
- Inspect raw response: `print(response.text)`

## Related Examples

- **openai_pooling_client.py:** Generic pooling API for embeddings
- **cohere_rerank_client.py:** Document reranking example
- Classification models in vLLM documentation

## References

- **vLLM API Server:** Documentation on classification endpoints
- **OpenAI API:** Compatible API design patterns
- **Requests Library:** [Documentation](https://requests.readthedocs.io/)
- **Classification Models:** HuggingFace model hub
