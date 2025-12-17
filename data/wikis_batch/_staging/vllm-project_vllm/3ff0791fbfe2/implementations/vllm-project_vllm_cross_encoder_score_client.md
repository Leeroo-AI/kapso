# Implementation: Cross-Encoder Scoring API Client

**File:** `/tmp/praxium_repo_583nq7ea/examples/pooling/score/openai_cross_encoder_score.py`
**Type:** HTTP Client Example
**Lines of Code:** 63
**Last Updated:** 2025-12-17

## Overview

The `openai_cross_encoder_score.py` script demonstrates how to use vLLM's `/score` endpoint for computing semantic similarity scores between text pairs using cross-encoder models. This client showcases the flexibility of the scoring API with three different input patterns for various reranking scenarios.

### Purpose

Provides a reference implementation for consuming vLLM's cross-encoder scoring service, illustrating flexible input formats for single-pair, one-to-many, and batch scoring operations.

### Key Features

- **Multiple Input Patterns**: Single pairs, one-to-many, and batch processing
- **Simple HTTP Interface**: Standard REST API with JSON payloads
- **Pretty-Printed Output**: Human-readable response formatting
- **Configurable Endpoint**: Parameterized host, port, and model selection

## Architecture

### Client-Server Communication

```
┌─────────────┐         HTTP POST         ┌──────────────┐
│   Client    │ ───────────────────────> │ vLLM Server  │
│             │    /score endpoint        │              │
│  This File  │                           │ (pooling)    │
│             │ <─────────────────────── │              │
└─────────────┘    JSON Response          └──────────────┘
```

### HTTP Request Function

```python
def post_http_request(prompt: dict, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    response = requests.post(api_url, headers=headers, json=prompt)
    return response
```

**Design Characteristics:**
- Simple wrapper around `requests.post`
- Custom User-Agent for identification
- Direct JSON serialization
- Synchronous request-response pattern

## Usage Patterns

### Pattern 1: Single Text Pair Scoring

Compute similarity between one query and one document:

```python
text_1 = "What is the capital of Brazil?"
text_2 = "The capital of Brazil is Brasilia."

prompt = {
    "model": "BAAI/bge-reranker-v2-m3",
    "text_1": text_1,
    "text_2": text_2
}

score_response = post_http_request(prompt=prompt, api_url=api_url)
```

**Response Format:**
```json
{
  "score": 0.95,
  "model": "BAAI/bge-reranker-v2-m3"
}
```

**Use Cases:**
- Single document relevance check
- Pairwise similarity computation
- Interactive query-document matching

### Pattern 2: One-to-Many Scoring

Score one query against multiple candidate documents:

```python
text_1 = "What is the capital of France?"
text_2 = [
    "The capital of Brazil is Brasilia.",
    "The capital of France is Paris."
]

prompt = {
    "model": "BAAI/bge-reranker-v2-m3",
    "text_1": text_1,
    "text_2": text_2
}

score_response = post_http_request(prompt=prompt, api_url=api_url)
```

**Response Format:**
```json
{
  "scores": [0.12, 0.98],
  "model": "BAAI/bge-reranker-v2-m3"
}
```

**Use Cases:**
- Document reranking
- Top-K candidate selection
- Search result relevance scoring

**Efficiency:** Single forward pass with batched candidates for improved throughput.

### Pattern 3: Batch Pairwise Scoring

Process multiple query-document pairs simultaneously:

```python
text_1 = [
    "What is the capital of Brazil?",
    "What is the capital of France?"
]
text_2 = [
    "The capital of Brazil is Brasilia.",
    "The capital of France is Paris."
]

prompt = {
    "model": "BAAI/bge-reranker-v2-m3",
    "text_1": text_1,
    "text_2": text_2
}

score_response = post_http_request(prompt=prompt, api_url=api_url)
```

**Response Format:**
```json
{
  "scores": [0.96, 0.98],
  "model": "BAAI/bge-reranker-v2-m3"
}
```

**Pairing:** Element-wise matching: `text_1[i]` paired with `text_2[i]`

**Use Cases:**
- Bulk relevance assessment
- Parallel query processing
- Evaluation pipelines

## Implementation Details

### Command-Line Arguments

```python
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="BAAI/bge-reranker-v2-m3")
    return parser.parse_args()
```

**Configuration Options:**
- `--host`: Server hostname (default: localhost)
- `--port`: Server port (default: 8000)
- `--model`: Model identifier for scoring

### Main Execution Flow

```python
def main(args):
    api_url = f"http://{args.host}:{args.port}/score"
    model_name = args.model

    # Pattern 1: Single pair
    print("\nPrompt when text_1 and text_2 are both strings:")
    pprint.pprint(prompt)
    print("\nScore Response:")
    pprint.pprint(score_response.json())

    # Pattern 2: One-to-many
    print("\nPrompt when text_1 is string and text_2 is a list:")
    # ... similar pattern ...

    # Pattern 3: Batch pairs
    print("\nPrompt when text_1 and text_2 are both lists:")
    # ... similar pattern ...
```

**Flow Characteristics:**
- Sequential demonstration of all patterns
- Clear output separation with headers
- Pretty-printed prompts and responses
- Educational structure for learning

## Server Requirements

### Starting vLLM Server

```bash
# Start server with pooling runner
vllm serve BAAI/bge-reranker-v2-m3 --runner pooling
```

**Key Configuration:**
- `--runner pooling`: Enables scoring/reranking mode
- Model must be a sequence classification model
- Server exposes `/score` endpoint automatically

### Compatible Models

The `/score` endpoint works with:
- **BGE Rerankers**: `BAAI/bge-reranker-v2-m3`, `BAAI/bge-reranker-base`
- **Cross-Encoders**: Models with sequence classification heads
- **Converted Models**: Output from `convert_model_to_seq_cls.py`

## API Specification

### Request Format

```json
{
  "model": "string (model identifier)",
  "text_1": "string | list[string]",
  "text_2": "string | list[string]"
}
```

**Validation Rules:**
- If both are lists: Must have equal length
- If one is list: Other must be string (broadcast pattern)
- If both strings: Single pair scoring

### Response Format

**Single Score:**
```json
{
  "score": float,
  "model": "string"
}
```

**Multiple Scores:**
```json
{
  "scores": [float, ...],
  "model": "string"
}
```

### HTTP Status Codes

- `200`: Success
- `400`: Invalid request format
- `500`: Server error

## Example Output

### Running the Client

```bash
$ python openai_cross_encoder_score.py --host localhost --port 8000

Prompt when text_1 and text_2 are both strings:
{'model': 'BAAI/bge-reranker-v2-m3',
 'text_1': 'What is the capital of Brazil?',
 'text_2': 'The capital of Brazil is Brasilia.'}

Score Response:
{'model': 'BAAI/bge-reranker-v2-m3', 'score': 0.9523}

Prompt when text_1 is string and text_2 is a list:
{'model': 'BAAI/bge-reranker-v2-m3',
 'text_1': 'What is the capital of France?',
 'text_2': ['The capital of Brazil is Brasilia.',
            'The capital of France is Paris.']}

Score Response:
{'model': 'BAAI/bge-reranker-v2-m3', 'scores': [0.1245, 0.9811]}

Prompt when text_1 and text_2 are both lists:
{'model': 'BAAI/bge-reranker-v2-m3',
 'text_1': ['What is the capital of Brazil?',
            'What is the capital of France?'],
 'text_2': ['The capital of Brazil is Brasilia.',
            'The capital of France is Paris.']}

Score Response:
{'model': 'BAAI/bge-reranker-v2-m3', 'scores': [0.9523, 0.9811]}
```

## Integration Patterns

### Document Reranking Pipeline

```python
# 1. Get initial candidates from retrieval
candidates = vector_search(query, top_k=100)

# 2. Rerank with cross-encoder
text_1 = query
text_2 = [doc.text for doc in candidates]

prompt = {"model": model_name, "text_1": text_1, "text_2": text_2}
response = post_http_request(prompt, api_url)

# 3. Sort by scores
scores = response.json()["scores"]
reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
```

### Parallel Query Processing

```python
# Batch multiple queries
queries = ["query1", "query2", "query3"]
documents = ["doc1", "doc2", "doc3"]

prompt = {"model": model_name, "text_1": queries, "text_2": documents}
response = post_http_request(prompt, api_url)
```

## Performance Considerations

### Batching Strategies

**One-to-Many Pattern:**
- Single query encoding
- Batched document encoding
- Most efficient for candidate reranking

**Batch Pairs Pattern:**
- All pairs processed in parallel
- GPU utilization improved
- Best for bulk processing

### Latency vs Throughput

| Pattern | Latency | Throughput | Use Case |
|---------|---------|------------|----------|
| Single Pair | Lowest | Low | Interactive |
| One-to-Many | Medium | High | Reranking |
| Batch Pairs | Higher | Highest | Bulk Processing |

## Error Handling

The example includes basic response checking:

```python
if response.status_code == 200:
    print("Request successful!")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"Request failed with status code: {response.status_code}")
    print(response.text)
```

**Production Considerations:**
- Add retry logic for transient failures
- Implement timeout handling
- Validate response structure
- Log failed requests

## Dependencies

```python
import argparse  # Command-line argument parsing
import pprint    # Pretty-printing responses
import requests  # HTTP client library
```

## Related Components

- **vLLM Score Endpoint**: Server-side implementation
- **Model Conversion Tool**: Prepares models for scoring
- **OpenAI Reranker Client**: Alternative API format
- **Pooling Runner**: vLLM component for classification tasks

## Best Practices

1. **Use One-to-Many for Reranking**: Most efficient for document reranking
2. **Batch When Possible**: Improves GPU utilization
3. **Monitor Response Times**: Track latency for different batch sizes
4. **Handle Errors Gracefully**: Implement proper error handling in production
5. **Cache Results**: Consider caching scores for repeated queries

## References

- **Source File**: `examples/pooling/score/openai_cross_encoder_score.py`
- **API Endpoint**: `/score`
- **Server Command**: `vllm serve <model> --runner pooling`
- **Compatible Models**: BGE-reranker series, cross-encoder models
- **Repository**: https://github.com/vllm-project/vllm

---

*This client demonstrates vLLM's flexible cross-encoder scoring API for semantic similarity and document reranking tasks.*
