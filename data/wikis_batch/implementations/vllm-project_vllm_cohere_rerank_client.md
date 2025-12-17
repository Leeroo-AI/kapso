# Cohere SDK Reranking Client Example

**Source:** `examples/pooling/score/cohere_rerank_client.py`
**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)
**Lines:** 47

## Overview

This example demonstrates vLLM's API compatibility with the Cohere Python SDK for document reranking. It shows how to use popular third-party SDKs with self-hosted vLLM servers, enabling users to swap Cohere's hosted service with their own infrastructure without changing client code.

## Implementation Pattern

### Architecture Design

**Client Compatibility Layer:**
- Uses official Cohere Python SDK
- Points SDK to local vLLM server
- API-compatible endpoints and response formats
- Supports both Cohere Client v1 and ClientV2

**Use Case: Document Reranking:**
Given a query and a set of documents, rerank the documents by relevance to the query. This is commonly used for:
- Search result optimization
- Information retrieval pipelines
- Question answering systems
- Document recommendation

### API Compatibility

**Key Benefit:**
Drop-in replacement for Cohere's API:
- No client code changes needed
- Same request/response format
- Compatible with existing integrations
- Cost savings through self-hosting

## Technical Implementation

### 1. Model and Data Setup

```python
import cohere
from cohere import Client, ClientV2

model = "BAAI/bge-reranker-base"

query = "What is the capital of France?"

documents = [
    "The capital of France is Paris",
    "Reranking is fun!",
    "vLLM is an open-source framework for fast AI serving",
]
```

**Model Selection:**
`BAAI/bge-reranker-base`:
- Popular reranking model from Beijing Academy of AI
- Cross-encoder architecture
- Fine-tuned specifically for document reranking
- Good balance of speed and accuracy

**Example Data:**
- Query: User's search or question
- Documents: Candidate documents to rank
- Model will score each document's relevance to the query

### 2. Generic Reranking Function

```python
def cohere_rerank(
    client: Client | ClientV2, model: str, query: str, documents: list[str]
) -> dict:
    return client.rerank(model=model, query=query, documents=documents)
```

**Purpose:**
Generic function that works with both Cohere SDK versions (v1 and v2).

**Parameters:**
- `client`: Cohere client instance (v1 or v2)
- `model`: Model identifier for reranking
- `query`: Search query or question
- `documents`: List of documents to rerank

**Returns:**
Dictionary with reranked results including relevance scores.

### 3. Cohere SDK v1 Usage

```python
def main():
    # cohere v1 client
    cohere_v1 = cohere.Client(base_url="http://localhost:8000", api_key="sk-fake-key")
    rerank_v1_result = cohere_rerank(cohere_v1, model, query, documents)
    print("-" * 50)
    print("rerank_v1_result:\n", rerank_v1_result)
    print("-" * 50)
```

**Client Configuration:**

**base_url:**
- Points to local vLLM server
- Default Cohere URL is overridden
- Can be any vLLM server address

**api_key:**
- Required by SDK (but not validated by vLLM)
- Use any placeholder value
- vLLM ignores authentication by default

### 4. Cohere SDK v2 Usage

```python
# or the v2
cohere_v2 = cohere.ClientV2("sk-fake-key", base_url="http://localhost:8000")
rerank_v2_result = cohere_rerank(cohere_v2, model, query, documents)
print("rerank_v2_result:\n", rerank_v2_result)
print("-" * 50)
```

**ClientV2 Differences:**
- Different constructor signature (API key first)
- Same rerank API interface
- Enhanced features in newer SDK version
- Both versions work with vLLM

## Response Format

### Expected Response Structure

```python
{
    "id": "rerank-request-id",
    "results": [
        {
            "index": 0,
            "relevance_score": 0.98,
            "document": {
                "text": "The capital of France is Paris"
            }
        },
        {
            "index": 2,
            "relevance_score": 0.15,
            "document": {
                "text": "vLLM is an open-source framework for fast AI serving"
            }
        },
        {
            "index": 1,
            "relevance_score": 0.05,
            "document": {
                "text": "Reranking is fun!"
            }
        }
    ],
    "meta": {
        "api_version": {"version": "1"}
    }
}
```

**Response Fields:**

**id:**
Unique request identifier.

**results:**
Array of reranked documents, sorted by relevance (high to low).

**results[].index:**
Original index of the document in input array.

**results[].relevance_score:**
Relevance score (0-1, higher = more relevant).
- Cross-encoder model outputs similarity scores
- Normalized to 0-1 range

**results[].document:**
Document content and metadata.

**meta:**
API version and metadata information.

## Usage Requirements

### Server Setup

Start vLLM with a reranking model:

```bash
vllm serve BAAI/bge-reranker-base
```

**Server Configuration:**

**Default Port:**
vLLM serves on port 8000 by default.

**Model Loading:**
The model must support reranking/scoring tasks:
- Cross-encoder models
- Models with scoring heads
- Trained for text pair classification

**Additional Options:**
```bash
vllm serve BAAI/bge-reranker-base \
  --port 8000 \
  --host 0.0.0.0 \
  --trust-remote-code
```

### Client Dependencies

```bash
pip install cohere
```

**Cohere SDK Installation:**
The example requires the official Cohere Python SDK:
- Supports v1 and v2 APIs
- Available via PyPI
- Regular updates and maintenance

**Version Compatibility:**
```python
# Check installed version
import cohere
print(cohere.__version__)
```

### Running the Example

```bash
# Install dependencies
pip install cohere

# Start vLLM server (in separate terminal)
vllm serve BAAI/bge-reranker-base

# Run client
python cohere_rerank_client.py
```

## Reranking Models

### Supported Models

**BAAI BGE Series:**
- `BAAI/bge-reranker-base`: Balanced performance
- `BAAI/bge-reranker-large`: Higher accuracy
- `BAAI/bge-reranker-v2-m3`: Multilingual support

**Other Rerankers:**
- `cross-encoder/ms-marco-MiniLM-L-12-v2`: MS MARCO trained
- `cross-encoder/ms-marco-TinyBERT-L-6`: Faster inference
- Custom fine-tuned rerankers

### Model Characteristics

**Cross-Encoder Architecture:**
- Jointly encodes query and document
- Outputs relevance score directly
- More accurate than bi-encoder retrieval
- Slower than embedding-based methods

**Input Format:**
- Query: Single text string
- Document: Single text string
- Model processes `[query, document]` pairs

**Output:**
- Scalar relevance score per pair
- Higher score = more relevant

## Production Patterns

### Error Handling

```python
import cohere
from cohere.errors import CohereAPIError

def rerank_with_error_handling(
    client: cohere.Client,
    model: str,
    query: str,
    documents: list[str]
) -> dict:
    """Rerank with comprehensive error handling."""
    try:
        result = client.rerank(
            model=model,
            query=query,
            documents=documents
        )
        return result

    except CohereAPIError as e:
        print(f"API error: {e}")
        print(f"Status code: {e.status_code}")
        print(f"Message: {e.message}")
        raise

    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
```

### Batch Reranking

```python
def rerank_multiple_queries(
    client: cohere.Client,
    model: str,
    queries: list[str],
    documents: list[str]
) -> list[dict]:
    """Rerank documents for multiple queries."""
    results = []

    for query in queries:
        try:
            result = client.rerank(
                model=model,
                query=query,
                documents=documents
            )
            results.append({
                "query": query,
                "reranked": result,
                "success": True
            })
        except Exception as e:
            results.append({
                "query": query,
                "error": str(e),
                "success": False
            })

    return results
```

### Top-K Filtering

```python
def rerank_top_k(
    client: cohere.Client,
    model: str,
    query: str,
    documents: list[str],
    top_k: int = 5
) -> list[dict]:
    """Rerank and return only top-k results."""
    result = client.rerank(
        model=model,
        query=query,
        documents=documents,
        top_n=top_k  # Cohere SDK parameter
    )

    # Extract top results
    top_results = [
        {
            "text": item.document.text,
            "score": item.relevance_score,
            "rank": i + 1
        }
        for i, item in enumerate(result.results)
    ]

    return top_results
```

### Score Threshold Filtering

```python
def rerank_with_threshold(
    client: cohere.Client,
    model: str,
    query: str,
    documents: list[str],
    threshold: float = 0.5
) -> list[dict]:
    """Rerank and filter by score threshold."""
    result = client.rerank(
        model=model,
        query=query,
        documents=documents
    )

    # Filter by threshold
    filtered_results = [
        {
            "text": item.document.text,
            "score": item.relevance_score,
            "index": item.index
        }
        for item in result.results
        if item.relevance_score >= threshold
    ]

    return filtered_results
```

## Integration Examples

### Search Pipeline

```python
class SearchEngine:
    """Search engine with reranking."""

    def __init__(self, vllm_url, model_name):
        self.client = cohere.Client(
            base_url=vllm_url,
            api_key="sk-fake-key"
        )
        self.model = model_name

    def search(self, query, initial_candidates, top_k=10):
        """Search with reranking."""
        # Initial retrieval (e.g., from vector DB)
        # initial_candidates = vector_search(query, top_k=100)

        # Rerank candidates
        rerank_result = self.client.rerank(
            model=self.model,
            query=query,
            documents=initial_candidates,
            top_n=top_k
        )

        # Format results
        results = [
            {
                "document": item.document.text,
                "score": item.relevance_score,
                "rank": i + 1
            }
            for i, item in enumerate(rerank_result.results)
        ]

        return results
```

### RAG Pipeline

```python
class RAGSystem:
    """Retrieval-Augmented Generation with reranking."""

    def __init__(self, vllm_rerank_url, llm_client):
        self.rerank_client = cohere.Client(
            base_url=vllm_rerank_url,
            api_key="sk-fake-key"
        )
        self.llm_client = llm_client

    def answer_question(self, question, knowledge_base):
        """Answer question using RAG with reranking."""
        # Step 1: Retrieve candidate documents
        candidates = self.retrieve_candidates(question, knowledge_base)

        # Step 2: Rerank candidates
        rerank_result = self.rerank_client.rerank(
            model="BAAI/bge-reranker-base",
            query=question,
            documents=candidates,
            top_n=3
        )

        # Step 3: Extract top documents
        context_docs = [
            item.document.text
            for item in rerank_result.results
        ]

        # Step 4: Generate answer with context
        context = "\n\n".join(context_docs)
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        answer = self.llm_client.generate(prompt)

        return {
            "answer": answer,
            "sources": context_docs,
            "relevance_scores": [
                item.relevance_score
                for item in rerank_result.results
            ]
        }
```

### Question Answering

```python
def find_best_answer(
    client: cohere.Client,
    question: str,
    candidate_answers: list[str]
) -> dict:
    """Find best answer from candidates."""
    result = client.rerank(
        model="BAAI/bge-reranker-base",
        query=question,
        documents=candidate_answers
    )

    # Get top answer
    best_result = result.results[0]

    return {
        "question": question,
        "answer": best_result.document.text,
        "confidence": best_result.relevance_score
    }

# Example usage
question = "What is the capital of France?"
candidates = [
    "Paris is the capital city of France",
    "France is a country in Europe",
    "The Eiffel Tower is in Paris"
]

answer = find_best_answer(cohere_client, question, candidates)
print(f"Best answer: {answer['answer']}")
print(f"Confidence: {answer['confidence']:.2f}")
```

## Performance Considerations

### Latency Optimization

**Model Size Trade-offs:**
- `bge-reranker-base`: 100-200ms per batch
- `bge-reranker-large`: 200-400ms per batch
- Smaller models: Faster but less accurate

**Batch Processing:**
```python
# Process all query-document pairs in single call
result = client.rerank(
    model=model,
    query=query,
    documents=documents  # All documents in one call
)
```

**Avoid Sequential Calls:**
```python
# Bad: Sequential calls
for doc in documents:
    result = client.rerank(query=query, documents=[doc])

# Good: Single batch call
result = client.rerank(query=query, documents=documents)
```

### Caching Strategies

```python
import hashlib
from functools import lru_cache

def cache_key(query: str, documents: tuple[str]) -> str:
    """Generate cache key for rerank request."""
    content = query + "".join(documents)
    return hashlib.md5(content.encode()).hexdigest()

@lru_cache(maxsize=1000)
def rerank_cached(
    query: str,
    documents: tuple[str],  # Tuple for hashability
    client_id: str
) -> dict:
    """Rerank with caching."""
    # Note: client must be stored separately
    client = get_client(client_id)

    result = client.rerank(
        model="BAAI/bge-reranker-base",
        query=query,
        documents=list(documents)
    )

    return result

# Usage
result = rerank_cached(
    "What is the capital of France?",
    tuple(documents),  # Convert to tuple
    "default_client"
)
```

### Parallel Processing

```python
import concurrent.futures

def rerank_parallel(
    queries: list[str],
    documents: list[str],
    max_workers: int = 4
) -> list[dict]:
    """Rerank multiple queries in parallel."""
    client = cohere.Client(
        base_url="http://localhost:8000",
        api_key="sk-fake-key"
    )

    def rerank_single(query):
        return client.rerank(
            model="BAAI/bge-reranker-base",
            query=query,
            documents=documents
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(rerank_single, queries))

    return results
```

## Comparison with Hosted Cohere

### Cost Savings

**Cohere Hosted (as of 2024):**
- ~$0.002 per 1K rerank requests
- 100K requests/month = $200

**Self-Hosted vLLM:**
- One-time GPU cost
- No per-request charges
- Break-even after ~100M requests

### Performance Comparison

**Latency:**
- Hosted: Network latency + processing
- Self-hosted: Processing only (if local)

**Throughput:**
- Hosted: Rate-limited by plan
- Self-hosted: Limited by hardware

**Privacy:**
- Hosted: Data sent to Cohere
- Self-hosted: Data stays on-premise

## Troubleshooting

### Common Issues

**Connection Refused:**
```
cohere.errors.CohereAPIError: Connection refused
```

**Solution:**
- Verify vLLM server is running
- Check server URL and port
- Test with curl: `curl http://localhost:8000/health`

**Model Not Found:**
```
{"error": "Model not found"}
```

**Solution:**
- Ensure model name matches server
- Check vLLM server logs
- Verify model supports reranking

**SDK Version Incompatibility:**
```
TypeError: __init__() got unexpected keyword argument
```

**Solution:**
- Check Cohere SDK version: `pip show cohere`
- Update SDK: `pip install --upgrade cohere`
- Review example for correct usage

**Invalid API Key (if authentication enabled):**
```
{"error": "Invalid API key"}
```

**Solution:**
- If vLLM has authentication, use valid key
- Otherwise, any placeholder works
- Check server configuration

## Related Examples

- **openai_pooling_client.py:** Generic pooling endpoint
- **openai_classification_client.py:** Classification example
- Reranking models in vLLM documentation

## References

- **Cohere Python SDK:** [GitHub](https://github.com/cohere-ai/cohere-python)
- **BGE Reranker Models:** [HuggingFace](https://huggingface.co/BAAI)
- **vLLM API Compatibility:** Documentation on API endpoints
- **Cross-Encoders:** Reranking architecture explanation
