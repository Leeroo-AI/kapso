# File: `examples/online_serving/openai_completion_client.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 53 |
| Functions | `parse_args`, `main` |
| Imports | argparse, openai |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** OpenAI completions API client

**Mechanism:** Basic example using OpenAI's legacy completions endpoint (not chat) with vLLM. Generates multiple completions (n=2) from a prompt with logprobs. Supports both streaming and non-streaming modes.

**Significance:** Shows compatibility with OpenAI's original completion API format, useful for applications that use raw text completion rather than structured chat messages.
