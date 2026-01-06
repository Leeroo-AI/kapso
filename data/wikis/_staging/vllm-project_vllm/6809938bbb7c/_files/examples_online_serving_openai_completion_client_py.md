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

**Purpose:** OpenAI Completions API example (non-chat models)

**Mechanism:** Uses the /v1/completions endpoint (not /v1/chat/completions) for raw text completion. Demonstrates parameters like echo, n (number of completions), logprobs, and streaming. Sends a raw prompt string rather than structured messages. Suitable for base models without chat templates.

**Significance:** Shows vLLM's compatibility with OpenAI's legacy Completions API. Important for applications using base models or requiring multiple completion candidates. Demonstrates parameter options (logprobs for token probabilities, echo for prompt repetition) useful for advanced use cases like uncertainty estimation or prompt engineering.
