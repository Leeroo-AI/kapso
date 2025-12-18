# File: `examples/pooling/token_embed/jina_embeddings_v4.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 71 |
| Functions | `get_embeddings` |
| Imports | torch, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Multi-vector embedding example for Jina v4

**Mechanism:** Demonstrates token-level embeddings with Jina-embeddings-v4 supporting both text and vision inputs. Creates prompts in multiple languages (German, Japanese) and with images, uses `pooling_task="token_embed"` to get per-token embeddings, then implements custom pooling logic that: (1) identifies vision tokens using special token IDs, (2) extracts vision token embeddings for images, (3) uses all tokens for text, (4) applies mean pooling and normalization.

**Significance:** Advanced example showing late-interaction retrieval patterns where individual token embeddings are preserved for more fine-grained similarity computation. Critical for ColBERT-style multi-vector retrieval systems and multimodal search applications that need token-level matching granularity.
