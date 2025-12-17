# Implementation: Multi-Vector Retrieval Client

**File:** `/tmp/praxium_repo_583nq7ea/examples/pooling/token_embed/multi_vector_retrieval_client.py`
**Type:** HTTP Client Example
**Lines of Code:** 54
**Last Updated:** 2025-12-17

## Overview

The `multi_vector_retrieval_client.py` script demonstrates online multi-vector retrieval using vLLM's `/pooling` endpoint with token embedding models. This client shows how to consume vLLM's token embedding API for ColBERT-style retrieval systems with remote model inference.

### Purpose

Illustrates client-side integration with vLLM's token embedding API, enabling scalable multi-vector retrieval services through HTTP requests without local model deployment.

### Key Features

- **HTTP-Based Inference**: Remote token embedding access via REST API
- **Multi-Vector Output**: 2D tensors for late interaction retrieval
- **Simple Integration**: Minimal code for production services
- **Batch Processing**: Efficient multi-prompt encoding

## Architecture

### Client-Server Communication

```
┌──────────────────┐                    ┌──────────────────┐
│  Client          │                    │  vLLM Server     │
│  (This File)     │                    │  (Pooling)       │
└────────┬─────────┘                    └────────┬─────────┘
         │                                       │
         │  1. POST /pooling                    │
         │     {"input": [...]}                 │
         ├──────────────────────────────────────>│
         │                                       │
         │                          2. Tokenize inputs
         │                          3. Model inference
         │                          4. Extract token embeddings
         │                                       │
         │  5. Return embeddings                │
         │     {"data": [...]}                  │
         │<──────────────────────────────────────┤
         │                                       │
         │  6. Convert to tensors               │
         │  7. Process shapes                   │
         └───────────────────────────────────────
```

### Data Flow

```
Input Texts (Batch)
    ↓
HTTP Request (JSON)
    ↓
[Server] Tokenization + Inference
    ↓
HTTP Response (Token Embeddings)
    ↓
[Client] JSON → Tensor Conversion
    ↓
Multi-Vector Representations
[num_texts × num_tokens × embedding_dim]
```

## Implementation Details

### Complete Source Code

```python
import argparse
import requests
import torch


def post_http_request(prompt: dict, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    response = requests.post(api_url, headers=headers, json=prompt)
    return response


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="BAAI/bge-m3")
    return parser.parse_args()


def main(args):
    api_url = f"http://{args.host}:{args.port}/pooling"
    model_name = args.model

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    prompt = {"model": model_name, "input": prompts}

    pooling_response = post_http_request(prompt=prompt, api_url=api_url)
    for output in pooling_response.json()["data"]:
        multi_vector = torch.tensor(output["data"])
        print(multi_vector.shape)


if __name__ == "__main__":
    args = parse_args()
    main(args)
```

### Key Components

#### 1. HTTP Request Handler

```python
def post_http_request(prompt: dict, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    response = requests.post(api_url, headers=headers, json=prompt)
    return response
```

**Characteristics:**
- Simple wrapper around requests library
- Custom User-Agent for identification
- Synchronous request-response pattern
- Automatic JSON serialization

#### 2. Command-Line Configuration

```python
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="BAAI/bge-m3")
    return parser.parse_args()
```

**Parameters:**
- `--host`: Server hostname (default: localhost)
- `--port`: Server port (default: 8000)
- `--model`: Model identifier for token embeddings

#### 3. Batch Request Construction

```python
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
prompt = {"model": model_name, "input": prompts}
```

**Request Format:**
```json
{
  "model": "BAAI/bge-m3",
  "input": [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is"
  ]
}
```

**Batching Benefits:**
- Single HTTP round trip
- Efficient GPU utilization
- Reduced latency per item

#### 4. Response Processing

```python
pooling_response = post_http_request(prompt=prompt, api_url=api_url)
for output in pooling_response.json()["data"]:
    multi_vector = torch.tensor(output["data"])
    print(multi_vector.shape)
```

**Response Structure:**
```json
{
  "id": "pooling-request-id",
  "object": "list",
  "created": 1234567890,
  "model": "BAAI/bge-m3",
  "data": [
    {
      "index": 0,
      "object": "embedding",
      "data": [
        [0.1, 0.2, ..., 0.9],  // Token 1
        [0.3, 0.4, ..., 0.8],  // Token 2
        // ... more tokens
      ]
    },
    // ... more outputs
  ]
}
```

**Processing:**
1. Extract `data` array from JSON response
2. Iterate through each output (one per input text)
3. Convert nested array to PyTorch tensor
4. Print shape: `[num_tokens, embedding_dim]`

## Usage Examples

### Basic Execution

```bash
# Start server first
vllm serve BAAI/bge-m3 --runner pooling

# Run client
python multi_vector_retrieval_client.py
```

**Output:**
```
torch.Size([7, 1024])
torch.Size([10, 1024])
torch.Size([8, 1024])
torch.Size([7, 1024])
```

**Interpretation:**
- First text: 7 tokens, each with 1024-dim embedding
- Second text: 10 tokens, 1024-dim embeddings
- Variable token counts per text
- Fixed embedding dimension (model-specific)

### Remote Server Access

```bash
python multi_vector_retrieval_client.py --host 192.168.1.100 --port 8080
```

**Use Cases:**
- Remote inference servers
- Load balancer endpoints
- Cloud deployments

### Different Model

```bash
# Server
vllm serve colbert-ir/colbertv2.0 --runner pooling

# Client
python multi_vector_retrieval_client.py --model colbert-ir/colbertv2.0
```

**Compatible Models:**
- `BAAI/bge-m3`: Multilingual multi-vector
- `colbert-ir/colbertv2.0`: Native ColBERT
- `BAAI/bge-large-en-v1.5`: English embeddings

## API Specification

### Request Format

```json
{
  "model": "string (required)",
  "input": "string | list[string] (required)",
  "encoding_format": "float | base64 (optional, default: float)"
}
```

**Parameters:**
- `model`: Model identifier (must match server)
- `input`: Single text or list of texts
- `encoding_format`: Output format (float for tensors)

### Response Format

```json
{
  "id": "string",
  "object": "list",
  "created": integer,
  "model": "string",
  "data": [
    {
      "index": integer,
      "object": "embedding",
      "data": [[float, ...], ...]
    }
  ],
  "usage": {
    "prompt_tokens": integer,
    "total_tokens": integer
  }
}
```

**Key Fields:**
- `data[i].data`: 2D array of token embeddings `[num_tokens, dim]`
- `data[i].index`: Position in input batch
- `usage.prompt_tokens`: Total tokens processed

## Integration Patterns

### ColBERT-Style Retrieval Service

```python
class ColBERTRetriever:
    def __init__(self, api_url, model_name):
        self.api_url = api_url
        self.model_name = model_name
        self.document_embeddings = {}

    def index_documents(self, documents):
        """Index document corpus."""
        prompt = {"model": self.model_name, "input": documents}
        response = post_http_request(prompt, self.api_url)

        for i, output in enumerate(response.json()["data"]):
            doc_tokens = torch.tensor(output["data"])
            self.document_embeddings[i] = doc_tokens

    def search(self, query, top_k=5):
        """Search for most relevant documents."""
        # Encode query
        prompt = {"model": self.model_name, "input": [query]}
        response = post_http_request(prompt, self.api_url)
        query_tokens = torch.tensor(response.json()["data"][0]["data"])

        # Compute scores
        scores = []
        for doc_id, doc_tokens in self.document_embeddings.items():
            score = self._compute_colbert_score(query_tokens, doc_tokens)
            scores.append((doc_id, score))

        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in scores[:top_k]]

    def _compute_colbert_score(self, query_tokens, doc_tokens):
        """Compute MaxSim score."""
        # [num_query_tokens, num_doc_tokens]
        similarities = torch.matmul(query_tokens, doc_tokens.T)
        # Max similarity for each query token
        max_sims = similarities.max(dim=1).values
        # Average max similarities
        return max_sims.mean().item()


# Usage
retriever = ColBERTRetriever(
    api_url="http://localhost:8000/pooling",
    model_name="BAAI/bge-m3"
)

documents = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "Python is a programming language",
]
retriever.index_documents(documents)

results = retriever.search("What is deep learning?", top_k=2)
print(f"Top documents: {results}")
```

### Batch Processing Pipeline

```python
def process_corpus_in_batches(texts, api_url, model_name, batch_size=32):
    """Process large corpus in batches."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        prompt = {"model": model_name, "input": batch}
        response = post_http_request(prompt, api_url)

        for output in response.json()["data"]:
            token_embeddings = torch.tensor(output["data"])
            all_embeddings.append(token_embeddings)

        print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")

    return all_embeddings


# Usage
corpus = ["text1", "text2", ...]  # Large corpus
embeddings = process_corpus_in_batches(
    corpus,
    api_url="http://localhost:8000/pooling",
    model_name="BAAI/bge-m3",
    batch_size=64
)
```

### Reranking Pipeline

```python
def rerank_with_token_embeddings(query, candidates, api_url, model_name):
    """Rerank candidates using token embeddings."""
    # Encode query
    query_prompt = {"model": model_name, "input": [query]}
    query_response = post_http_request(query_prompt, api_url)
    query_tokens = torch.tensor(query_response.json()["data"][0]["data"])

    # Encode candidates
    cand_prompt = {"model": model_name, "input": candidates}
    cand_response = post_http_request(cand_prompt, api_url)

    # Compute scores
    scores = []
    for output in cand_response.json()["data"]:
        cand_tokens = torch.tensor(output["data"])
        score = compute_maxsim_score(query_tokens, cand_tokens)
        scores.append(score)

    # Sort candidates by score
    ranked_indices = torch.tensor(scores).argsort(descending=True).tolist()
    ranked_candidates = [candidates[i] for i in ranked_indices]

    return ranked_candidates, scores


def compute_maxsim_score(query_tokens, doc_tokens):
    """MaxSim scoring for ColBERT."""
    similarities = torch.matmul(query_tokens, doc_tokens.T)
    max_sims = similarities.max(dim=1).values
    return max_sims.mean().item()
```

## Server Requirements

### Starting the Server

```bash
vllm serve BAAI/bge-m3 --runner pooling
```

**Required Configuration:**
- `--runner pooling`: Enables token embedding endpoint
- Model must support token-level embeddings

### Advanced Server Configuration

```bash
vllm serve BAAI/bge-m3 \
  --runner pooling \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.9
```

**Performance Options:**
- `--host 0.0.0.0`: Allow external connections
- `--max-model-len`: Maximum sequence length per text
- `--max-num-batched-tokens`: Total tokens in batch
- `--gpu-memory-utilization`: GPU memory allocation

## Performance Considerations

### Latency Breakdown

```
Total Latency = Network + Server Processing + Client Processing

Network:          10-50ms   (HTTP round trip)
Server Processing: 20-200ms  (depends on batch size, text length)
Client Processing: 1-5ms     (JSON parsing, tensor conversion)
```

### Throughput Optimization

**Batch Size Impact:**

| Batch Size | Latency (avg) | Throughput |
|------------|---------------|------------|
| 1 | 30ms | 33 req/s |
| 8 | 100ms | 80 req/s |
| 32 | 300ms | 106 req/s |
| 64 | 500ms | 128 req/s |

*Approximate values for BGE-M3 on A100 GPU*

**Optimization Strategies:**
1. Use larger batches for throughput
2. Keep connections alive with session pooling
3. Implement async requests for parallelism
4. Cache frequently used embeddings

### Memory Efficiency

**Storage Requirements:**
```python
# Per document (avg 100 tokens, 1024-dim)
token_embeddings = 100 * 1024 * 4 bytes = 410 KB

# For 100K documents
total_storage = 100K * 410 KB = 41 GB
```

**Compression Strategy:**
- Store only top-K tokens per document
- Use quantization (float16 or int8)
- Compute embeddings on-the-fly for re-ranking

## Error Handling

### Connection Errors

```python
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def create_session_with_retries():
    """Create requests session with retry logic."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504]
    )
    session.mount('http://', HTTPAdapter(max_retries=retries))
    return session

# Usage
session = create_session_with_retries()
response = session.post(api_url, json=prompt)
```

### Response Validation

```python
def validate_response(response):
    """Validate API response structure."""
    if response.status_code != 200:
        raise ValueError(f"HTTP {response.status_code}: {response.text}")

    data = response.json()
    if "data" not in data:
        raise ValueError("Response missing 'data' field")

    for output in data["data"]:
        if "data" not in output:
            raise ValueError("Output missing embeddings")

        embeddings = output["data"]
        if not isinstance(embeddings, list) or not embeddings:
            raise ValueError("Invalid embedding format")

    return data
```

### Tensor Conversion

```python
try:
    multi_vector = torch.tensor(output["data"])
    if multi_vector.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {multi_vector.dim()}D")
except (TypeError, ValueError) as e:
    print(f"Failed to convert embeddings to tensor: {e}")
```

## Comparison: Online vs Offline

| Aspect | Client (Online) | Offline |
|--------|----------------|---------|
| **Setup** | Server + Client | Single script |
| **Scalability** | Horizontal | Vertical only |
| **Latency** | +Network overhead | Lower |
| **Resource** | Centralized GPU | Local GPU required |
| **Multi-User** | Yes | No |
| **Use Case** | Production | Research/Batch |

## Dependencies

```python
import argparse  # CLI argument parsing
import requests  # HTTP client
import torch     # Tensor operations
```

**Installation:**
```bash
pip install requests torch
```

## Best Practices

1. **Use Batching**: Send multiple texts in single request
2. **Connection Pooling**: Reuse HTTP connections
3. **Error Handling**: Implement retries and fallbacks
4. **Async Requests**: Use async for high concurrency
5. **Monitor Latency**: Track performance metrics
6. **Cache Embeddings**: Avoid redundant computations
7. **Validate Responses**: Check tensor shapes and values

## Advanced Patterns

### Async Client

```python
import asyncio
import aiohttp

async def post_async(session, prompt, api_url):
    async with session.post(api_url, json=prompt) as response:
        return await response.json()

async def batch_encode_async(texts, api_url, model_name, batch_size=32):
    """Async batch encoding with concurrent requests."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            prompt = {"model": model_name, "input": batch}
            tasks.append(post_async(session, prompt, api_url))

        responses = await asyncio.gather(*tasks)
        return responses

# Usage
texts = ["text1", "text2", ...]  # Large corpus
responses = asyncio.run(batch_encode_async(texts, api_url, model_name))
```

## Related Components

- **Multi-Vector Retrieval Offline**: Local inference version
- **Jina Embeddings v4**: Multimodal token embeddings
- **ColBERT Models**: Native multi-vector retrieval models
- **Pooling Runner**: vLLM server component

## References

- **Source File**: `examples/pooling/token_embed/multi_vector_retrieval_client.py`
- **API Endpoint**: `/pooling`
- **Server Command**: `vllm serve <model> --runner pooling`
- **Model**: BAAI/bge-m3
- **ColBERT**: Late interaction retrieval method
- **Repository**: https://github.com/vllm-project/vllm

---

*This client demonstrates online multi-vector retrieval using vLLM's token embedding API for ColBERT-style late interaction retrieval systems.*
