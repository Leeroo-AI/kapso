# File: `benchmarks/kernels/benchmark_reshape_and_cache.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 172 |
| Functions | `run_benchmark`, `main` |
| Imports | random, tabulate, time, torch, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarks reshape_and_cache kernel that stores newly computed KV tensors into paged KV cache.

**Mechanism:** Measures performance of reshaping key/value tensors and caching them in block-based storage across different sequence lengths and cache configurations.

**Significance:** Critical operation in attention mechanism that impacts prefill performance. Optimizing this kernel improves overall throughput.
