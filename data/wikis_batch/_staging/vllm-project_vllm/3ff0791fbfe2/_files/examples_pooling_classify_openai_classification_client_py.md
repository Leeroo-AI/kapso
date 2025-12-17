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

**Purpose:** OpenAI classification API client

**Mechanism:** This HTTP client demonstrates text classification using vLLM's `/classify` endpoint. It sends a batch of text prompts to a classification model server and receives predicted class labels. The client constructs a payload with the model name and input texts, posts it to the API, and prints the classification results.

**Significance:** Example demonstrating how to consume vLLM's classification API for text categorization tasks. Shows the expected request format and response structure for classification models like Qwen2.5-1.5B-apeach.
