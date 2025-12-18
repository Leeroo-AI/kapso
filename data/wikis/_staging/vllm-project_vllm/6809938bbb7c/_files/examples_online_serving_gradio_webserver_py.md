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

**Purpose:** Gradio web interface for vLLM's legacy API server

**Mechanism:** Creates a simple Gradio Blocks interface with text input/output boxes. Connects to the legacy /generate endpoint using raw HTTP requests (not OpenAI API). Streams responses using Server-Sent Events and yields incremental outputs to update the UI in real-time.

**Significance:** Legacy example for the older vllm.entrypoints.api_server. Less sophisticated than the OpenAI-based version (gradio_openai_chatbot_webserver.py) but demonstrates basic streaming text completion with Gradio. Useful for understanding the legacy API but not recommended for new projects.
