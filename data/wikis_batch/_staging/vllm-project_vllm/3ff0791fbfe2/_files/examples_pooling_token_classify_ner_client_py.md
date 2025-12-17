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

**Purpose:** Named entity recognition client

**Mechanism:** HTTP client for online NER using vLLM's `/pooling` endpoint with token classification models. It sends text to the server, receives per-token logits in the response, applies argmax to determine predicted labels, and uses the tokenizer to map predictions back to entity types. The client handles tokenization locally to align predictions with input tokens.

**Significance:** Example showing how to consume vLLM's token classification API for online NER services. Demonstrates the client-side processing needed to interpret token-level model outputs for structured extraction tasks.
