# File: `benchmarks/kernels/benchmark_trtllm_prefill_attention.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 305 |
| Functions | `to_float8`, `benchmark_prefill`, `write_results_to_csv` |
| Imports | csv, datetime, flashinfer, os, torch, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarks TRT-LLM prefill attention with FlashInfer

**Mechanism:** Benchmarks FlashInfer single_prefill_with_kv_cache for prefill phase attention. Tests FP8 KV cache scenarios with optional FP8 conversion. Sweeps num_kv_heads (8/16/32), head_dim (128/256), num_qo_heads, seq_len (128-8192), and page_size (8/16). Creates random query and KV cache data, runs prefill attention with CUDA event timing. Saves results to timestamped CSV with configuration and latency data. Supports warmup iterations and profiler integration. Tests both causal and non-causal attention modes.

**Significance:** Validates FlashInfer prefill attention performance for TRT-LLM compatibility. Prefill attention dominates latency for long prompts and is critical for time-to-first-token (TTFT). Benchmark evaluates FP8 KV cache impact on prefill throughput. Essential for understanding prefill scaling characteristics and optimal page_size selection. Key for high-throughput batch prefill scenarios.
