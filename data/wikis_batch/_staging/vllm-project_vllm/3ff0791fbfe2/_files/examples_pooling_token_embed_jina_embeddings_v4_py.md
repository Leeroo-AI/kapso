# File: `examples/pooling/token_embed/jina_embeddings_v4.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 71 |
| Functions | `get_embeddings` |
| Imports | torch, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Jina multimodal token embeddings

**Mechanism:** This script demonstrates token-level embeddings with Jina Embeddings v4, which supports both text and vision inputs. It creates prompts for multilingual text (German, Japanese) and an image, then uses `llm.encode()` with `pooling_task="token_embed"` to get per-token embeddings. The `get_embeddings()` function handles special processing: for images, it extracts only vision tokens between special markers; for text, it uses all tokens. Each embedding is mean-pooled and normalized.

**Significance:** Example showing advanced multimodal token embedding extraction with vision-language models. Demonstrates handling of vision-specific tokens and multilingual text embeddings for fine-grained retrieval applications.
