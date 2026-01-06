# File: `examples/pooling/score/cohere_rerank_client.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 47 |
| Functions | `cohere_rerank`, `main` |
| Imports | cohere |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Cohere SDK compatibility example for reranking

**Mechanism:** Tests vLLM's OpenAI-compatible rerank API using both Cohere SDK v1 and v2 clients. Connects to local vLLM server with a reranking model (`BAAI/bge-reranker-base`), sends query-document pairs, and demonstrates API compatibility with Cohere's reranking interface. Uses fake API key since authentication is not required for local testing.

**Significance:** Shows that vLLM's rerank endpoint is compatible with the popular Cohere SDK, allowing easy migration of existing Cohere-based applications to vLLM. Important for developers seeking cost-effective or privacy-focused alternatives to hosted reranking APIs.
