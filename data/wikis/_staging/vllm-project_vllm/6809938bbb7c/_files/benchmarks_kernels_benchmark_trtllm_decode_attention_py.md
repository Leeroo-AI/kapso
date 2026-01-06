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

**Purpose:** Benchmarks TensorRT-LLM decode attention implementation for autoregressive generation.

**Mechanism:** Measures decode (single-token) attention performance with varying KV cache sizes, comparing TensorRT-LLM kernels against vLLM implementations.

**Significance:** Decode attention is the performance bottleneck in generation. Evaluating TensorRT-LLM integration helps identify optimization opportunities.
