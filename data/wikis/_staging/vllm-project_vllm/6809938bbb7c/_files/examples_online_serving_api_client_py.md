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

**Purpose:** Demonstrates basic HTTP client for vLLM's legacy demo API server

**Mechanism:** Uses Python requests library to send prompts to /generate endpoint, supports both streaming and non-streaming responses with configurable parameters (temperature, max_tokens, beam search). Includes terminal display utilities to clear lines for streaming output visualization.

**Significance:** Example client for the legacy vllm.entrypoints.api_server (NOT for production). Shows basic integration patterns but the documentation explicitly recommends using the OpenAI-compatible API (vllm serve) for production workloads. Useful for quick demos and simple benchmarking.
