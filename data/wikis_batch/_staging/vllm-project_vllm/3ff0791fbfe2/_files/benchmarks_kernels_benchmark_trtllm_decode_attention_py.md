# File: `benchmarks/kernels/benchmark_trtllm_decode_attention.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 290 |
| Functions | `to_float8`, `benchmark_decode`, `write_results_to_csv` |
| Imports | csv, datetime, flashinfer, os, torch, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarks TRT-LLM decode attention with FlashInfer

**Mechanism:** Benchmarks FlashInfer batch_decode_with_shared_prefix_paged_kv_cache for decode phase attention. Tests FP8 KV cache scenarios with optional FP8 conversion using to_float8. Sweeps num_kv_heads (8/16/32), head_dim (128/256), batch_size (1-512), and page_size (8/16). Creates random KV cache data and runs decode attention with CUDA event timing. Saves results to timestamped CSV including configuration details and latency measurements. Supports warmup iterations and profiler integration.

**Significance:** Validates FlashInfer decode attention performance for TRT-LLM compatibility. Decode attention is critical for generation phase latency in inference. Benchmark helps understand FP8 KV cache impact on decode performance. Essential for choosing optimal page_size and validating FlashInfer integration. Important for low-latency serving scenarios.
