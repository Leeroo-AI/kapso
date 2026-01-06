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

**Purpose:** HTTP client for cross-encoder scoring API

**Mechanism:** Tests vLLM's `/score` endpoint with three input patterns: (1) single text pair (string, string), (2) query with multiple candidates (string, list), (3) paired lists of equal length. Demonstrates batch scoring capabilities with BGE-reranker model, showing how to compute relevance scores between queries and documents.

**Significance:** Reference implementation for using vLLM's scoring endpoint for information retrieval and reranking tasks. Shows flexible input formats allowing both pairwise scoring and efficient batch processing of multiple query-document pairs.
