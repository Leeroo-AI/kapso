# File: `examples/pooling/classify/openai_classification_client.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 53 |
| Functions | `post_http_request`, `parse_args`, `main` |
| Imports | argparse, pprint, requests |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** HTTP client example for testing vLLM classification API endpoint

**Mechanism:** Sends classification requests to a vLLM server via POST to the `/classify` endpoint. Demonstrates the OpenAI-compatible classification API with sample prompts and prints results using pprint. Requires a classification model server to be running (e.g., `jason9693/Qwen2.5-1.5B-apeach`).

**Significance:** Example client demonstrating how to interact with vLLM's classification endpoint for text categorization tasks. Shows the expected request format (model name + input texts) and response structure for classification inference.
