# File: `benchmarks/fused_kernels/layernorm_rms_benchmarks.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 310 |
| Classes | `bench_params_t` |
| Functions | `get_bench_params`, `unfused_int8_impl`, `unfused_fp8_impl`, `unfused_groupwise_fp8_impl`, `fused_impl`, `fused_groupwise_impl`, `bench_fn`, `bench`, `... +2 more` |
| Imports | collections, dataclasses, itertools, pickle, time, torch, tqdm, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmark fused RMSNorm+quantization kernels

**Mechanism:** Compares unfused (RMSNorm then quantize) vs fused implementations for INT8/FP8 quantization with optional residual addition. Tests per-token and per-token-group (blockwise) quantization across various batch sizes (1-2048), hidden dimensions (1K-8K), dtypes (bf16/float), and group sizes (64/128). Uses PyTorch benchmark utilities with min 1s runtime.

**Significance:** Performance validation for fused kernel optimization. Demonstrates benefits of combining RMSNorm and dynamic quantization into single kernel, reducing memory bandwidth and improving latency. Critical for efficient quantized inference pipelines.
