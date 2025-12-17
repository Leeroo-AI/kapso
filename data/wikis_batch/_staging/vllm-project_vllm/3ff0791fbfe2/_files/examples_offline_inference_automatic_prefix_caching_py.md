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

**Purpose:** Demonstrates Automatic Prefix Caching (APC) performance benefits.

**Mechanism:** Runs inference twice with shared long prompt prefix (markdown table): first without APC as baseline, then with enable_prefix_caching=True. Measures and compares generation times for prompts sharing the same prefix to show KV cache reuse improvements. Uses cleanup_dist_env_and_memory between runs.

**Significance:** Example demonstrating APC feature that reduces redundant computation by caching and reusing KV pairs from shared prompt prefixes.
