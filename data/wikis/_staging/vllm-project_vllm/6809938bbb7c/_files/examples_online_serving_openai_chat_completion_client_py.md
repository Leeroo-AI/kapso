# File: `examples/online_serving/openai_chat_completion_client.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 64 |
| Functions | `parse_args`, `main` |
| Imports | argparse, openai |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Basic OpenAI chat completion client example

**Mechanism:** Simple demonstration of using the OpenAI Python SDK to connect to vLLM's /v1 endpoint. Makes a chat completion request with multi-turn conversation context, supports optional streaming flag. Automatically discovers the first available model from the server.

**Significance:** Fundamental starting point for OpenAI API compatibility. Shows the minimal code needed to integrate vLLM with existing OpenAI-based applications. Critical reference for developers migrating from OpenAI to vLLM - demonstrates API parity and drop-in replacement capability.
