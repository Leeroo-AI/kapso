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

**Purpose:** Generic HTTP client for testing pooling API

**Mechanism:** Sends requests to vLLM's `/pooling` endpoint demonstrating two input formats: (1) Completions API format with simple text strings, and (2) Chat API format with message arrays containing role/content structures. Tests with a reward model (`internlm/internlm2-1_8b-reward`) and prints JSON responses.

**Significance:** Reference implementation showing the dual API compatibility of vLLM's pooling endpoint. Demonstrates that pooling models can accept both simple text inputs and structured chat messages, making them flexible for various embedding and scoring tasks.
