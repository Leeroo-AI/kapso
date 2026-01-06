# File: `examples/offline_inference/prefix_caching.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 98 |
| Functions | `main` |
| Imports | vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates manual prefix caching using prompt_token_ids parameter to explicitly control KV cache reuse.

**Mechanism:** Generates completions with shared prompt prefixes by passing prompt_token_ids directly. vLLM detects matching prefix token sequences and reuses their KV cache computations. Compares performance between first request (computes full prefix) and subsequent requests (reuses cached prefix).

**Significance:** Shows explicit control over prefix caching for advanced optimization scenarios. Differs from automatic_prefix_caching.py by demonstrating manual token ID management for precise cache control.
