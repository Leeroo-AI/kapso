# File: `examples/offline_inference/automatic_prefix_caching.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 103 |
| Functions | `get_generation_time`, `main` |
| Imports | time, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates Automatic Prefix Caching (APC) to reuse KV cache from shared prompt prefixes, improving inference speed.

**Mechanism:** Uses a long markdown table as a shared prefix across multiple prompts. Compares generation time between requests that share the same prefix, showing how vLLM automatically caches and reuses KV pairs from common prompt segments. Enables APC via `enable_prefix_caching=True`.

**Significance:** Illustrates performance optimization technique for scenarios with repeated prompt prefixes (e.g., system prompts, long contexts). Critical for understanding how vLLM reduces redundant computation through KV cache reuse.
