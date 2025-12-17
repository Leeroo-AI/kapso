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

**Purpose:** Cohere SDK reranking example

**Mechanism:** This client demonstrates vLLM's compatibility with the Cohere Python SDK for document reranking. It shows usage with both Cohere Client v1 and ClientV2, pointing them at a local vLLM server instead of Cohere's API. The script sends a query and documents to be reranked, with the model scoring document relevance to return them in order of importance.

**Significance:** Example proving vLLM's API compatibility with popular third-party SDKs. Enables users to swap Cohere's hosted reranking service with self-hosted vLLM models without changing client code.
