# File: `examples/pooling/score/openai_reranker.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 42 |
| Functions | `main` |
| Imports | json, requests |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** OpenAI-compatible reranking client

**Mechanism:** Simple HTTP client that uses vLLM's `/rerank` endpoint compatible with Jina and Cohere's reranking APIs. It sends a query and list of documents to the server, which returns ranked results with relevance scores. The endpoint follows the OpenAI-style API conventions for easy integration with existing tools.

**Significance:** Minimal example demonstrating vLLM's OpenAI-compatible reranking endpoint. Enables drop-in replacement of commercial reranking services with self-hosted models using standard API formats.
