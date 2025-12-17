# File: `examples/online_serving/gradio_webserver.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 75 |
| Functions | `http_bot`, `build_demo`, `parse_args`, `main` |
| Imports | argparse, gradio, json, requests |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Gradio UI for legacy API

**Mechanism:** Creates a simple Gradio web interface for text completion using vLLM's legacy API server. Makes direct HTTP requests to the `/generate` endpoint and streams responses back to the UI. Provides a basic text input/output interface.

**Significance:** Example showing Gradio integration with vLLM's legacy (non-OpenAI-compatible) API. Simpler than the OpenAI-based version but uses the demonstration-only API server.
