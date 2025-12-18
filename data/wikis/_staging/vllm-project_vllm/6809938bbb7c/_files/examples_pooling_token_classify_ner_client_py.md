# File: `examples/pooling/token_classify/ner_client.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 71 |
| Functions | `post_http_request`, `parse_args`, `main` |
| Imports | argparse, requests, torch |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Online NER client using pooling endpoint

**Mechanism:** HTTP client for token classification via vLLM server's `/pooling` endpoint. Sends text to NeuroBERT-NER model, receives per-token logits as JSON, converts to tensor, applies argmax to get predictions, and maps to entity labels using HuggingFace tokenizer and config. Demonstrates online inference for NER tasks.

**Significance:** Shows how to perform token-level classification through vLLM's API server, enabling NER as a service. Useful for building production NER systems with vLLM backend, handling tokenization alignment between client and server.
