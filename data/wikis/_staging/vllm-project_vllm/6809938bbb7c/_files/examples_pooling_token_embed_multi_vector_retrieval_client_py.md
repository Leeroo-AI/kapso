# File: `examples/pooling/token_embed/multi_vector_retrieval_client.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 54 |
| Functions | `post_http_request`, `parse_args`, `main` |
| Imports | argparse, requests, torch |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Online multi-vector embedding client

**Mechanism:** HTTP client requesting token-level embeddings from vLLM's `/pooling` endpoint. Sends batch of texts to BGE-M3 model, receives JSON with per-token embedding matrices, converts to tensors, and displays shapes showing [sequence_length, embedding_dim] format.

**Significance:** Shows how to obtain multi-vector embeddings through vLLM's API for online retrieval applications. Enables building distributed late-interaction retrieval systems where embedding service is separate from indexing/search infrastructure.
