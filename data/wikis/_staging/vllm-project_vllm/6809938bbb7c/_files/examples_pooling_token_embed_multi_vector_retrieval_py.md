# File: `examples/pooling/token_embed/multi_vector_retrieval.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 56 |
| Functions | `parse_args`, `main` |
| Imports | argparse, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Offline multi-vector embedding generation

**Mechanism:** Uses vLLM to generate token-level embeddings with BGE-M3 model. Calls both `llm.embed()` for standard pooled embeddings and `llm.encode(pooling_task="token_embed")` for multi-vector representations. Prints embedding dimensions showing the difference between single-vector and multi-vector outputs.

**Significance:** Demonstrates dual-mode embedding generation where the same model can produce both pooled (single vector per text) and token-level (matrix per text) representations. Important for understanding the flexibility of modern embedding models and choosing appropriate granularity for different retrieval tasks.
