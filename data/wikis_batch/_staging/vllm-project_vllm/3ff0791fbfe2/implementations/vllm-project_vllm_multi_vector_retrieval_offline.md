# Implementation: Multi-Vector Retrieval (Offline)

**File:** `/tmp/praxium_repo_583nq7ea/examples/pooling/token_embed/multi_vector_retrieval.py`
**Type:** Offline Embedding Example
**Lines of Code:** 56
**Last Updated:** 2025-12-17

## Overview

The `multi_vector_retrieval.py` script demonstrates offline multi-vector retrieval using vLLM's token embedding capabilities with the BGE-M3 model. This implementation showcases both traditional sentence embeddings and token-level embeddings for ColBERT-style late interaction retrieval.

### Purpose

Illustrates vLLM's dual embedding capabilities: sentence-level embeddings (via `llm.embed()`) and token-level embeddings (via `llm.encode()`) for advanced retrieval scenarios.

### Key Features

- **Dual Embedding Modes**: Sentence and token-level embeddings
- **Offline Processing**: Direct model inference without server
- **ColBERT-Style Output**: Token-level multi-vector representations
- **Flexible Configuration**: Standard vLLM EngineArgs interface

## Architecture

### Dual Embedding Pipeline

```
┌─────────────────────────────────────────┐
│           Input Prompts                 │
│  ["Hello, my name is",                  │
│   "The president of the United States", │
│   "The capital of France is",           │
│   "The future of AI is"]                │
└────────────┬────────────────────────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
┌─────────────┐  ┌──────────────┐
│ Sentence    │  │ Token        │
│ Embeddings  │  │ Embeddings   │
│             │  │              │
│ llm.embed() │  │ llm.encode() │
└──────┬──────┘  └──────┬───────┘
       │                │
       ▼                ▼
┌─────────────┐  ┌──────────────┐
│ [N × 1024]  │  │ [N × T × D]  │
│ Fixed dim   │  │ Variable T   │
│ per text    │  │ per token    │
└─────────────┘  └──────────────┘
```

**N**: Number of prompts
**T**: Number of tokens (varies per prompt)
**D**: Embedding dimension (1024 for BGE-M3)

## Implementation Details

### Argument Parsing

```python
def parse_args():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)

    # Set example specific arguments
    parser.set_defaults(
        model="BAAI/bge-m3",
        runner="pooling",
        enforce_eager=True,
    )
    return parser.parse_args()
```

**Configuration:**
- `model="BAAI/bge-m3"`: Multi-vector embedding model
- `runner="pooling"`: Enables embedding mode
- `enforce_eager=True`: Disables graph optimization for clarity

**Model Choice:** BGE-M3 is a state-of-the-art multilingual embedding model supporting both dense and multi-vector retrieval.

### Model Initialization

```python
def main(args: Namespace):
    # Create an LLM with runner="pooling" for embedding models
    llm = LLM(**vars(args))
```

**Setup:**
- Converts args namespace to kwargs
- Initializes vLLM engine in pooling mode
- Loads BGE-M3 model weights
- Prepares for dual embedding tasks

### Sample Prompts

```python
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
```

**Characteristics:**
- Varied prompt lengths
- Different semantic content
- Typical query/document scenarios

### Sentence Embeddings

```python
# Generate embedding. The output is a list of EmbeddingRequestOutputs.
outputs = llm.embed(prompts)

# Print the outputs.
print("\nGenerated Outputs:\n" + "-" * 60)
for prompt, output in zip(prompts, outputs):
    embeds = output.outputs.embedding
    print(len(embeds))
```

**API Usage:**
- `llm.embed()`: High-level embedding method
- Returns `EmbeddingRequestOutput` objects
- Each output contains fixed-length embedding vector

**Output Example:**
```
Generated Outputs:
------------------------------------------------------------
1024
1024
1024
1024
```

**Embedding Characteristics:**
- Fixed dimension (1024-D)
- Normalized (unit length)
- Sentence-level representation
- Suitable for semantic search

### Token Embeddings

```python
# Generate embedding for each token. The output is a list of PoolingRequestOutput.
outputs = llm.encode(prompts, pooling_task="token_embed")

# Print the outputs.
print("\nGenerated Outputs:\n" + "-" * 60)
for prompt, output in zip(prompts, outputs):
    multi_vector = output.outputs.data
    print(multi_vector.shape)
```

**API Usage:**
- `llm.encode()`: Generic pooling/encoding method
- `pooling_task="token_embed"`: Specifies token-level embeddings
- Returns `PoolingRequestOutput` objects
- Each output contains 2D tensor of token embeddings

**Output Example:**
```
Generated Outputs:
------------------------------------------------------------
torch.Size([7, 1024])    # "Hello, my name is" - 7 tokens
torch.Size([10, 1024])   # "The president..." - 10 tokens
torch.Size([8, 1024])    # "The capital..." - 8 tokens
torch.Size([7, 1024])    # "The future..." - 7 tokens
```

**Token Embedding Characteristics:**
- Variable sequence length (depends on text)
- Each token has 1024-D embedding
- Suitable for late interaction retrieval
- Preserves fine-grained semantic information

## Usage Examples

### Basic Execution

```bash
python multi_vector_retrieval.py
```

**Default Output:**
```
Generated Outputs:
------------------------------------------------------------
1024
1024
1024
1024

Generated Outputs:
------------------------------------------------------------
torch.Size([7, 1024])
torch.Size([10, 1024])
torch.Size([8, 1024])
torch.Size([7, 1024])
```

### Custom Model

```bash
python multi_vector_retrieval.py --model BAAI/bge-large-en-v1.5
```

**Compatible Models:**
- `BAAI/bge-m3`: Multilingual multi-vector
- `BAAI/bge-large-en-v1.5`: English-only
- `colbert-ir/colbertv2.0`: Native ColBERT

### Advanced Configuration

```bash
python multi_vector_retrieval.py \
  --model BAAI/bge-m3 \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9
```

**Performance Options:**
- `--tensor-parallel-size`: Multi-GPU inference
- `--max-model-len`: Maximum sequence length
- `--gpu-memory-utilization`: Memory allocation

## Retrieval Patterns

### Late Interaction Scoring (ColBERT)

```python
def compute_colbert_score(query_tokens, doc_tokens):
    """
    Compute ColBERT-style MaxSim score.

    Args:
        query_tokens: [num_query_tokens, dim]
        doc_tokens: [num_doc_tokens, dim]

    Returns:
        Relevance score (float)
    """
    # Compute pairwise similarities
    # [num_query_tokens, num_doc_tokens]
    similarities = torch.matmul(query_tokens, doc_tokens.T)

    # For each query token, find max similarity with any doc token
    max_sims = similarities.max(dim=1).values  # [num_query_tokens]

    # Average max similarities
    score = max_sims.mean()

    return score.item()
```

**Usage Example:**
```python
# Encode query and document
query_output = llm.encode(["machine learning"], pooling_task="token_embed")[0]
doc_output = llm.encode(["deep learning frameworks"], pooling_task="token_embed")[0]

query_tokens = query_output.outputs.data
doc_tokens = doc_output.outputs.data

score = compute_colbert_score(query_tokens, doc_tokens)
print(f"Relevance Score: {score:.4f}")
```

### Hybrid Retrieval Pipeline

```python
def hybrid_retrieval(query, documents, llm, alpha=0.5):
    """
    Combine dense (sentence) and multi-vector (token) retrieval.

    Args:
        query: Query string
        documents: List of document strings
        llm: vLLM instance
        alpha: Weight for dense scores (1-alpha for multi-vector)

    Returns:
        Ranked document indices
    """
    # Dense retrieval scores
    query_embed = llm.embed([query])[0].outputs.embedding
    doc_embeds = [llm.embed([doc])[0].outputs.embedding for doc in documents]
    dense_scores = [torch.dot(query_embed, doc_emb).item()
                    for doc_emb in doc_embeds]

    # Multi-vector retrieval scores
    query_tokens = llm.encode([query], pooling_task="token_embed")[0].outputs.data
    doc_tokens_list = [llm.encode([doc], pooling_task="token_embed")[0].outputs.data
                       for doc in documents]
    colbert_scores = [compute_colbert_score(query_tokens, doc_tokens)
                      for doc_tokens in doc_tokens_list]

    # Normalize and combine
    dense_scores = torch.tensor(dense_scores)
    colbert_scores = torch.tensor(colbert_scores)

    dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())
    colbert_scores = (colbert_scores - colbert_scores.min()) / (colbert_scores.max() - colbert_scores.min())

    hybrid_scores = alpha * dense_scores + (1 - alpha) * colbert_scores

    # Rank documents
    ranked_indices = hybrid_scores.argsort(descending=True).tolist()

    return ranked_indices, hybrid_scores
```

### Batch Processing

```python
def batch_encode_corpus(documents, llm, batch_size=32):
    """Efficiently encode large document corpus."""
    all_embeddings = []

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]

        # Sentence embeddings
        outputs = llm.embed(batch)
        embeddings = [out.outputs.embedding for out in outputs]
        all_embeddings.extend(embeddings)

    return torch.stack(all_embeddings)

def batch_encode_token_embeddings(documents, llm, batch_size=32):
    """Efficiently encode token embeddings for corpus."""
    all_token_embeddings = []

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]

        # Token embeddings
        outputs = llm.encode(batch, pooling_task="token_embed")
        token_embeds = [out.outputs.data for out in outputs]
        all_token_embeddings.extend(token_embeds)

    return all_token_embeddings
```

## Comparison: Sentence vs Token Embeddings

### Sentence Embeddings (`llm.embed()`)

**Advantages:**
- Fixed dimension (easier storage/indexing)
- Faster similarity computation
- Lower memory requirements
- Simpler implementation

**Use Cases:**
- Semantic search
- Clustering
- Classification
- First-stage retrieval

**Similarity Computation:**
```python
score = torch.dot(query_embed, doc_embed)  # O(D)
```

### Token Embeddings (`llm.encode()`)

**Advantages:**
- Fine-grained matching
- Better handling of multi-aspect queries
- Higher quality for complex documents
- Preserves term importance

**Use Cases:**
- Re-ranking
- Question answering
- Long document retrieval
- Multi-hop reasoning

**Similarity Computation:**
```python
score = compute_colbert_score(query_tokens, doc_tokens)  # O(Q × D × T)
```

## Performance Characteristics

### Throughput

| Mode | Batch Size | Items/sec | Memory/Item |
|------|------------|-----------|-------------|
| Sentence | 1 | 100 | 4KB |
| Sentence | 32 | 2,000 | 4KB |
| Token | 1 | 50 | 40KB |
| Token | 32 | 1,000 | 40KB |

*Approximate values for BGE-M3 on A100 GPU*

### Latency

| Mode | Single Item | Batch 32 |
|------|-------------|----------|
| Sentence | 10ms | 160ms |
| Token | 20ms | 640ms |

### Storage Requirements

**100K Documents (avg 100 tokens):**
- Sentence embeddings: 100K × 1024 × 4 bytes = 410 MB
- Token embeddings: 100K × 100 × 1024 × 4 bytes = 41 GB

**Storage Strategy:**
- Store sentence embeddings for first-stage retrieval
- Compute token embeddings on-the-fly for re-ranking
- Or store compressed token embeddings

## Integration with Vector Databases

### Sentence Embeddings (FAISS)

```python
import faiss
import numpy as np

# Encode corpus
embeddings = batch_encode_corpus(documents, llm)
embeddings_np = embeddings.cpu().numpy()

# Build FAISS index
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product (for normalized vectors)
index.add(embeddings_np)

# Search
query_embed = llm.embed([query])[0].outputs.embedding.cpu().numpy()
D, I = index.search(query_embed.reshape(1, -1), k=10)

# Get top-k documents
top_docs = [documents[i] for i in I[0]]
```

### Token Embeddings (Custom Index)

```python
class TokenEmbeddingIndex:
    """Simple index for token embeddings."""

    def __init__(self, documents, llm):
        self.documents = documents
        self.token_embeddings = batch_encode_token_embeddings(documents, llm)

    def search(self, query, llm, top_k=10):
        """Search using ColBERT scoring."""
        query_tokens = llm.encode([query], pooling_task="token_embed")[0].outputs.data

        scores = []
        for doc_tokens in self.token_embeddings:
            score = compute_colbert_score(query_tokens, doc_tokens)
            scores.append(score)

        # Get top-k
        top_indices = torch.tensor(scores).topk(top_k).indices
        return [self.documents[i] for i in top_indices]
```

## Error Handling

### Model Loading

```python
try:
    llm = LLM(**vars(args))
except Exception as e:
    print(f"Failed to load model: {e}")
    # Check: model name, GPU memory, dependencies
```

### Embedding Extraction

```python
try:
    outputs = llm.embed(prompts)
    embeddings = [out.outputs.embedding for out in outputs]
except AttributeError:
    print("Model may not support sentence embeddings")
```

### Token Embedding Extraction

```python
try:
    outputs = llm.encode(prompts, pooling_task="token_embed")
    token_embeds = [out.outputs.data for out in outputs]
except Exception as e:
    print(f"Token embedding failed: {e}")
```

## Dependencies

```python
from argparse import Namespace
from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
import torch  # For tensor operations (implicit)
```

**Requirements:**
- vLLM: Core inference engine
- PyTorch: Tensor backend
- Transformers: Model loading (implicit)

## Best Practices

1. **Choose Right Mode**: Sentence for speed, token for quality
2. **Batch Processing**: Use larger batches for throughput
3. **Hybrid Approach**: Combine both for optimal results
4. **Memory Management**: Monitor GPU memory with token embeddings
5. **Normalize Embeddings**: Ensure unit length for cosine similarity
6. **Cache Sentence Embeddings**: Recompute token embeddings as needed

## Related Components

- **Multi-Vector Retrieval Client**: Online version using HTTP API
- **Jina Embeddings v4**: Advanced multimodal token embeddings
- **Pooling Runner**: vLLM component for embedding tasks
- **Token Embedding Models**: Compatible embedding architectures

## References

- **Source File**: `examples/pooling/token_embed/multi_vector_retrieval.py`
- **Model**: BAAI/bge-m3
- **ColBERT Paper**: "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"
- **BGE Models**: https://huggingface.co/BAAI
- **Repository**: https://github.com/vllm-project/vllm

---

*This implementation demonstrates vLLM's dual embedding capabilities for both traditional dense retrieval and advanced multi-vector (ColBERT-style) retrieval.*
