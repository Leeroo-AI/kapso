# OpenAI Pooling API Client Example

**Source:** `examples/pooling/pooling/openai_pooling_client.py`
**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)
**Lines:** 63

## Overview

This example demonstrates how to use vLLM's generic pooling endpoint, which provides a flexible API for extracting embeddings or pooled representations from models. It showcases two input formats - Completions-style and Chat-style - making it compatible with different client workflows and use cases.

## Implementation Pattern

### Architecture Design

The client interacts with vLLM's `/pooling` endpoint which supports:

**Use Cases:**
- Reward model scoring for RLHF
- Sentence embeddings extraction
- Text representation learning
- Similarity computation
- Preference modeling

**Input Flexibility:**
- Completions API format: Direct text input
- Chat API format: Structured messages with roles

**Output:**
Pooled model representations suitable for downstream tasks like scoring, ranking, or embedding-based retrieval.

## Technical Implementation

### 1. HTTP Request Function

```python
def post_http_request(prompt: dict, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    response = requests.post(api_url, headers=headers, json=prompt)
    return response
```

**Purpose:**
Generic HTTP POST function for sending requests to the pooling endpoint.

**Parameters:**
- `prompt`: Dictionary containing input data and configuration
- `api_url`: Full URL to the pooling endpoint

**Returns:**
`requests.Response` object with pooled representations.

### 2. Command-Line Configuration

```python
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="internlm/internlm2-1_8b-reward")

    return parser.parse_args()
```

**Arguments:**

**--host:**
- Server hostname or IP
- Default: `localhost`
- Example: `--host vllm.example.com`

**--port:**
- API server port
- Default: `8000`
- Standard vLLM port

**--model:**
- Model identifier
- Default: `internlm/internlm2-1_8b-reward` (reward model)
- Must match server configuration

### 3. Completions-Style Input

```python
def main(args):
    api_url = f"http://{args.host}:{args.port}/pooling"
    model_name = args.model

    # Input like Completions API
    prompt = {"model": model_name, "input": "vLLM is great!"}
    pooling_response = post_http_request(prompt=prompt, api_url=api_url)
    print("-" * 50)
    print("Pooling Response:")
    pprint.pprint(pooling_response.json())
    print("-" * 50)
```

**Request Format:**
```json
{
  "model": "internlm/internlm2-1_8b-reward",
  "input": "vLLM is great!"
}
```

**Characteristics:**
- Simple string input
- Direct text processing
- Minimal structure
- Similar to OpenAI Completions API

**Use Cases:**
- Single-text embeddings
- Simple scoring tasks
- Quick experiments

### 4. Chat-Style Input

```python
# Input like Chat API
prompt = {
    "model": model_name,
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": "vLLM is great!"}],
        }
    ],
}
pooling_response = post_http_request(prompt=prompt, api_url=api_url)
print("Pooling Response:")
pprint.pprint(pooling_response.json())
print("-" * 50)
```

**Request Format:**
```json
{
  "model": "internlm/internlm2-1_8b-reward",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "vLLM is great!"
        }
      ]
    }
  ]
}
```

**Characteristics:**
- Structured message format
- Role-based content (user, assistant, system)
- Array of content items
- Compatible with chat models
- Similar to OpenAI Chat API

**Use Cases:**
- Conversational reward modeling
- Multi-turn dialogue scoring
- Context-aware embeddings

## Request Format Details

### Completions-Style Request

**Minimal Format:**
```json
{
  "model": "model-name",
  "input": "text"
}
```

**With Options:**
```json
{
  "model": "model-name",
  "input": "text",
  "temperature": 0.0,
  "max_tokens": 512
}
```

**Batch Format:**
```json
{
  "model": "model-name",
  "input": ["text1", "text2", "text3"]
}
```

### Chat-Style Request

**Single Message:**
```json
{
  "model": "model-name",
  "messages": [
    {
      "role": "user",
      "content": [{"type": "text", "text": "Hello"}]
    }
  ]
}
```

**Conversation:**
```json
{
  "model": "model-name",
  "messages": [
    {
      "role": "system",
      "content": [{"type": "text", "text": "You are helpful"}]
    },
    {
      "role": "user",
      "content": [{"type": "text", "text": "What is AI?"}]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "AI is..."}]
    },
    {
      "role": "user",
      "content": [{"type": "text", "text": "Tell me more"}]
    }
  ]
}
```

**Multimodal Content:**
```json
{
  "model": "model-name",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "https://..."}}
      ]
    }
  ]
}
```

## Response Format

### Expected Response Structure

```json
{
  "id": "pooling-request-id",
  "object": "pooling",
  "created": 1234567890,
  "model": "internlm/internlm2-1_8b-reward",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, -0.456, 0.789, ...],
      "index": 0
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

**Response Fields:**

**id:**
Unique identifier for the request.

**object:**
Response type (`"pooling"` or `"embedding"`).

**created:**
Unix timestamp of processing.

**model:**
Echo of the model name.

**data:**
Array of pooling results (one per input).

**data[].embedding:**
- Vector representation of the input
- Dimension depends on model
- Typically 768, 1024, or higher dimensions

**usage:**
Token usage statistics.

### Reward Model Response

For reward models specifically:

```json
{
  "id": "pooling-request-id",
  "object": "pooling",
  "model": "internlm/internlm2-1_8b-reward",
  "data": [
    {
      "object": "score",
      "score": 8.5,
      "index": 0
    }
  ]
}
```

**score Field:**
Scalar reward score for the input text, used in RLHF training.

## Usage Requirements

### Starting the vLLM Server

```bash
# For embedding models
vllm serve sentence-transformers/all-MiniLM-L6-v2

# For reward models
vllm serve internlm/internlm2-1_8b-reward --trust-remote-code

# With custom port
vllm serve model-name --port 8080
```

**Server Configuration:**

**--trust-remote-code:**
Required for models with custom code (like InternLM reward models).

**--port:**
Port to bind (default: 8000).

**Additional Options:**
- `--max-model-len`: Maximum sequence length
- `--tensor-parallel-size`: For large models
- `--gpu-memory-utilization`: Memory management

### Running the Client

```bash
# Basic usage
python openai_pooling_client.py

# Custom server
python openai_pooling_client.py --host vllm-server --port 8080

# Different model
python openai_pooling_client.py --model sentence-transformers/all-MiniLM-L6-v2
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

Minimal dependencies, Python 3.7+ compatible.

## Model Types

### Supported Models

**Reward Models:**
- `internlm/internlm2-1_8b-reward`: RLHF reward scoring
- `OpenAssistant/reward-model-deberta-v3-large`: OA reward model
- Custom fine-tuned reward models

**Embedding Models:**
- `sentence-transformers/all-MiniLM-L6-v2`: General embeddings
- `BAAI/bge-large-en-v1.5`: High-quality embeddings
- `intfloat/e5-large-v2`: Instruction-tuned embeddings

**Pooling Models:**
- Models with pooling layers
- Encoder-only architectures
- Models with classification heads

### Model Requirements

**Architecture:**
Must support pooling or embedding extraction:
- BERT-like models
- Sentence transformers
- Reward models with value heads

**Output:**
Produces fixed-size representations:
- Vector embeddings
- Scalar scores
- Pooled hidden states

## Production Patterns

### Error Handling

```python
def post_http_request_with_retry(prompt: dict, api_url: str, max_retries=3):
    """Post request with error handling and retries."""
    headers = {"User-Agent": "Test Client"}

    for attempt in range(max_retries):
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=prompt,
                timeout=30
            )
            response.raise_for_status()
            return response

        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt + 1}/{max_retries}")
            if attempt == max_retries - 1:
                raise

        except requests.exceptions.HTTPError as e:
            print(f"HTTP error: {e.response.status_code}")
            print(f"Response: {e.response.text}")
            raise

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if attempt == max_retries - 1:
                raise

        time.sleep(2 ** attempt)  # Exponential backoff
```

### Batch Processing

```python
def get_embeddings_batch(texts, api_url, model_name, batch_size=32):
    """Get embeddings for multiple texts in batches."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        prompt = {
            "model": model_name,
            "input": batch  # Array of texts
        }

        response = post_http_request(prompt, api_url)
        result = response.json()

        # Extract embeddings
        for item in result["data"]:
            all_embeddings.append(item["embedding"])

    return all_embeddings
```

### Async Processing

```python
import asyncio
import aiohttp

async def get_embedding_async(session, text, api_url, model_name):
    """Get embedding asynchronously."""
    prompt = {"model": model_name, "input": text}

    async with session.post(api_url, json=prompt) as response:
        result = await response.json()
        return result["data"][0]["embedding"]

async def get_embeddings_concurrent(texts, api_url, model_name):
    """Process multiple texts concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [
            get_embedding_async(session, text, api_url, model_name)
            for text in texts
        ]
        return await asyncio.gather(*tasks)

# Usage
embeddings = asyncio.run(
    get_embeddings_concurrent(texts, api_url, model_name)
)
```

## Integration Examples

### Semantic Search

```python
import numpy as np

class SemanticSearchEngine:
    """Semantic search using pooling API."""

    def __init__(self, api_url, model_name):
        self.api_url = api_url
        self.model_name = model_name
        self.document_embeddings = []
        self.documents = []

    def index_documents(self, documents):
        """Index documents by computing embeddings."""
        print(f"Indexing {len(documents)} documents...")

        embeddings = get_embeddings_batch(
            documents,
            self.api_url,
            self.model_name
        )

        self.document_embeddings = np.array(embeddings)
        self.documents = documents

    def search(self, query, top_k=5):
        """Search for most similar documents."""
        # Get query embedding
        prompt = {"model": self.model_name, "input": query}
        response = post_http_request(prompt, self.api_url)
        query_embedding = np.array(
            response.json()["data"][0]["embedding"]
        )

        # Compute cosine similarity
        similarities = np.dot(
            self.document_embeddings,
            query_embedding
        ) / (
            np.linalg.norm(self.document_embeddings, axis=1) *
            np.linalg.norm(query_embedding)
        )

        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = [
            {
                "document": self.documents[i],
                "similarity": similarities[i]
            }
            for i in top_indices
        ]

        return results
```

### RLHF Reward Scoring

```python
class RewardScorer:
    """Score model outputs for RLHF."""

    def __init__(self, api_url, model_name):
        self.api_url = api_url
        self.model_name = model_name

    def score_completion(self, prompt, completion):
        """Score a prompt-completion pair."""
        # Format as conversation
        request = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": completion}]
                }
            ]
        }

        response = post_http_request(request, self.api_url)
        result = response.json()

        return result["data"][0]["score"]

    def rank_completions(self, prompt, completions):
        """Rank multiple completions by reward score."""
        scores = [
            self.score_completion(prompt, comp)
            for comp in completions
        ]

        # Sort by score (descending)
        ranked = sorted(
            zip(completions, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked
```

### Sentence Similarity

```python
def compute_similarity(text1, text2, api_url, model_name):
    """Compute cosine similarity between two texts."""
    # Get embeddings
    prompt = {
        "model": model_name,
        "input": [text1, text2]
    }

    response = post_http_request(prompt, api_url)
    result = response.json()

    emb1 = np.array(result["data"][0]["embedding"])
    emb2 = np.array(result["data"][1]["embedding"])

    # Cosine similarity
    similarity = np.dot(emb1, emb2) / (
        np.linalg.norm(emb1) * np.linalg.norm(emb2)
    )

    return similarity

# Usage
similarity = compute_similarity(
    "vLLM is great!",
    "vLLM is awesome!",
    api_url,
    model_name
)
print(f"Similarity: {similarity:.3f}")
```

## Performance Optimization

### Connection Pooling

```python
import requests

# Create session for connection reuse
session = requests.Session()

# Configure session
adapter = requests.adapters.HTTPAdapter(
    pool_connections=10,
    pool_maxsize=20
)
session.mount('http://', adapter)
session.mount('https://', adapter)

# Reuse session
for text in texts:
    prompt = {"model": model_name, "input": text}
    response = session.post(api_url, json=prompt)
    process_response(response)
```

### Batch Size Tuning

```python
def optimal_batch_size(model_name):
    """Recommend batch size based on model."""
    if "large" in model_name.lower():
        return 16
    elif "base" in model_name.lower():
        return 32
    else:
        return 64

batch_size = optimal_batch_size(model_name)
embeddings = get_embeddings_batch(texts, api_url, model_name, batch_size)
```

## Troubleshooting

### Common Issues

**Model Not Found:**
```json
{"error": "Model not found"}
```

**Solution:**
- Verify model name matches server
- Check server logs for loaded model
- Ensure model supports pooling

**Invalid Input Format:**
```json
{"error": "Invalid input format"}
```

**Solution:**
- Check JSON structure
- Ensure `input` or `messages` field present
- Validate message format for chat-style

**Dimension Mismatch:**
```
ValueError: shapes not aligned
```

**Solution:**
- Ensure all embeddings from same model
- Check embedding dimensions are consistent
- Verify model hasn't changed between calls

## Related Examples

- **openai_classification_client.py:** Classification endpoint
- **cohere_rerank_client.py:** Reranking with Cohere SDK
- Embedding examples in vLLM documentation

## References

- **vLLM Pooling API:** Documentation on pooling endpoints
- **OpenAI Embeddings API:** Compatible API design
- **Sentence Transformers:** Popular embedding models
- **RLHF:** Reward modeling for reinforcement learning
