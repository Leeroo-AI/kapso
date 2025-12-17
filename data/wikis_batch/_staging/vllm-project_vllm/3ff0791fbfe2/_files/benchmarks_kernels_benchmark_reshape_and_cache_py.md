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

**Purpose:** Benchmarks reshape_and_cache KV operation

**Mechanism:** Tests reshape_and_cache kernel that converts key/value tensors from [T, H, D] format into paged KV cache blocks. Creates random slot mappings for tokens across KV cache blocks. Supports optional FP8 KV cache quantization with per-kernel scales. Tests both regular execution and CUDA graph modes. Measures latency across multiple token counts and reports results in formatted table using tabulate. Supports various head sizes (64-256) and KV cache dtypes (auto/fp8).

**Significance:** Critical for KV cache performance evaluation. The reshape_and_cache operation runs during prefill to populate the KV cache with new key/value tensors. Performance here directly impacts prefill latency. Validates FP8 KV cache quantization overhead. Essential for understanding KV cache write performance and ensuring it doesn't become a bottleneck in inference.
