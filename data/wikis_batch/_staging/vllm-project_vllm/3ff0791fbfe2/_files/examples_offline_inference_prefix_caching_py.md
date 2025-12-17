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

**Purpose:** Demonstrates manual prefix caching comparison

**Mechanism:** Creates two LLM instances (without and with enable_prefix_caching), runs same prompts with shared prefix, compares outputs and demonstrates speedup. Uses cleanup_dist_env_and_memory between instances. Includes warmup run for cached version.

**Significance:** Example showing prefix caching benefits through direct performance comparison with shared prompt prefixes.
