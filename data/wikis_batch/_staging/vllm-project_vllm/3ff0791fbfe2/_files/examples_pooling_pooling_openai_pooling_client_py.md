# File: `examples/pooling/pooling/openai_pooling_client.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 63 |
| Functions | `post_http_request`, `parse_args`, `main` |
| Imports | argparse, pprint, requests |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Generic pooling API client

**Mechanism:** This HTTP client demonstrates vLLM's generic `/pooling` endpoint for extracting model embeddings or pooled representations. It shows two input formats: (1) Completions-style with direct text input, and (2) Chat-style with structured messages containing role and content. Both formats are sent to the pooling endpoint which returns pooled model outputs suitable for reward models or embedding extraction.

**Significance:** Example showing the flexibility of vLLM's pooling API to handle multiple input formats. Useful for reward models like InternLM2-1.8B that need to score or embed conversational inputs.
