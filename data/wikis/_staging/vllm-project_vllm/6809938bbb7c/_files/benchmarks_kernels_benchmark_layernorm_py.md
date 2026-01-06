# File: `benchmarks/kernels/benchmark_layernorm.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 94 |
| Functions | `main` |
| Imports | time, torch, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarks RMSNorm (Root Mean Square Layer Normalization) kernel performance with and without residual connections across different precisions and tensor sizes.

**Mechanism:** Creates random input tensors and RMSNorm layers, runs warmup iterations followed by timed benchmarks measuring kernel execution time. Supports optional profiling with CUDA profiler and various data types (float16, bfloat16, float32).

**Significance:** RMSNorm is a critical normalization layer used in modern LLMs like LLaMA. Benchmarking helps optimize this frequently-called operation and validate performance across different hardware and precision modes.
