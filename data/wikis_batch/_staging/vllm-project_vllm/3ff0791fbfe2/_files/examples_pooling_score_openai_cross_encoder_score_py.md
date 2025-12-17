# File: `examples/pooling/score/openai_cross_encoder_score.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 63 |
| Functions | `post_http_request`, `parse_args`, `main` |
| Imports | argparse, pprint, requests |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Cross-encoder scoring API client

**Mechanism:** This HTTP client demonstrates vLLM's `/score` endpoint for cross-encoder models that compute relevance scores between text pairs. It shows three usage patterns: (1) single text pair scoring, (2) one-to-many scoring (one query, multiple candidates), and (3) batch pairwise scoring (matching lists). The client sends text_1 and text_2 parameters to receive numerical relevance scores.

**Significance:** Example for using vLLM's scoring API with cross-encoder rerankers like BGE-reranker-v2-m3. Shows the flexible input patterns supported for document ranking and semantic similarity tasks.
