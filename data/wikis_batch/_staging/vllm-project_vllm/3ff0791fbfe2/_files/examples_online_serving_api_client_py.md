# File: `examples/online_serving/api_client.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 93 |
| Functions | `clear_line`, `post_http_request`, `get_streaming_response`, `get_response`, `parse_args`, `main` |
| Imports | argparse, collections, json, requests |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demo HTTP client for vLLM API server

**Mechanism:** Provides a simple Python client that sends HTTP POST requests to vLLM's legacy API server (`vllm.entrypoints.api_server`). Supports both streaming and non-streaming text generation with the `/generate` endpoint. Includes utilities for formatting streaming responses with terminal escape codes.

**Significance:** Example code demonstrating basic HTTP integration with vLLM. Note: This uses the legacy API server for demonstration purposes only - production deployments should use `vllm serve` with the OpenAI-compatible API instead.
