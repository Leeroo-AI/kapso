# File: `benchmarks/kernels/benchmark_bitblas.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 244 |
| Imports | bitblas, packaging, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmark BitBLAS quantized kernels

**Mechanism:** Tests Microsoft's BitBLAS library for low-bit quantized matrix multiplication. Supports INT4/INT2/INT1/NF4/FP4 weight formats with configurable activation dtypes, accumulation types, group sizes, scaling, and zeros modes. Tests comprehensive shape sets covering square matrices and real model dimensions (BLOOM-176B, OPT-65B, LLAMA-70B) at batch sizes 1 and 8192. Requires BitBLAS>=0.0.1.dev14.

**Significance:** Third-party quantization library evaluation. BitBLAS provides TVM-based optimized kernels for extreme quantization. Essential for comparing vLLM's native quantization against specialized frameworks, particularly for sub-4-bit and non-uniform quantization formats.
