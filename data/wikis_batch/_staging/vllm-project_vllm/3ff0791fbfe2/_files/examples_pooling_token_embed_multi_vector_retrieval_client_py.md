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

**Purpose:** Multi-vector retrieval client

**Mechanism:** HTTP client for online multi-vector retrieval using vLLM's `/pooling` endpoint with token embedding models. It sends text prompts to the server and receives token-level embeddings in response. Each output is a 2D tensor (tokens × embedding_dim) that can be used for fine-grained semantic matching. The client reconstructs the embedding tensors from the API response for downstream processing.

**Significance:** Example demonstrating how to consume vLLM's token embedding API for ColBERT-style retrieval systems. Shows the expected response format for multi-vector representations used in advanced semantic search applications.
