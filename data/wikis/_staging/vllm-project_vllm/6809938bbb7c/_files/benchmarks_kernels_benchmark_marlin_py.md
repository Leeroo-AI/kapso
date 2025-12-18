# File: `benchmarks/kernels/benchmark_marlin.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 413 |
| Functions | `bench_run`, `main` |
| Imports | benchmark_shapes, torch, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarks Marlin quantized GEMM kernels including standard Marlin, Marlin 2:4 sparsity variant, GPTQ repack operations, and AllSpark W8A16 implementations across various quantization configurations.

**Mechanism:** Tests multiple quantization schemes (INT4, INT8, FP4, FP8) with different group sizes and activation ordering. Compares Marlin GEMM, Marlin 24 (2:4 sparsity), GPTQ repack, and AllSpark variants against PyTorch baseline. Supports act_order and is_k_full configurations.

**Significance:** Essential for evaluating Marlin's highly-optimized quantized GEMM implementations which are critical for efficient INT4/INT8 weight quantization. Covers various quantization variants including structured sparsity and different quantization strategies used in popular quantized models.
