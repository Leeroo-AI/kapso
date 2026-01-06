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

**Purpose:** Direct HTTP example for rerank endpoint

**Mechanism:** Simple HTTP POST to vLLM's `/rerank` endpoint using requests library. Sends query and document list to BGE-reranker model, receives ranked results with relevance scores. Compatible with Jina and Cohere rerank API format.

**Significance:** Minimal example showing raw HTTP interaction with rerank API without SDK dependencies. Useful for understanding the wire format and for integration with non-Python clients or custom HTTP libraries.
