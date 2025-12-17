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

**Purpose:** Multi-vector retrieval offline example

**Mechanism:** This script demonstrates offline multi-vector retrieval using vLLM's token embedding capability. It runs inference with BGE-M3 model using `llm.embed()` for sentence-level embeddings and `llm.encode()` with `pooling_task="token_embed"` for token-level embeddings. The token embeddings produce a matrix per input (tokens × dimensions) suitable for ColBERT-style late interaction retrieval.

**Significance:** Example showing vLLM's dual embedding capabilities: both traditional sentence embeddings and token-level multi-vector representations. Demonstrates configuration for models supporting late interaction retrieval methods.
