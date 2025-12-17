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

**Purpose:** Benchmarks Marlin quantized GEMM kernels

**Mechanism:** Comprehensive benchmark for Marlin (FP4/FP8/4-bit int) quantized matrix multiplication. Tests gptq_marlin_gemm, gptq_marlin_24_gemm (2:4 sparsity), gptq_marlin_repack, and allspark_w8a16_gemm across model weight shapes. Supports various configurations: act_order (activation reordering), is_k_full (full K dimension), different quant types (uint4/uint4b8/float4_e2m1f/float8), and group sizes (16/128/-1). Compares against PyTorch baseline matmul using torch.utils.benchmark. Tests both FP32 and FP16 accumulation modes.

**Significance:** Primary validation tool for Marlin kernel performance and correctness. Essential for evaluating Marlin across different quantization schemes, sparsity patterns, and matrix shapes. Critical for determining optimal Marlin configuration and validating performance improvements. Key component of quantization strategy evaluation for vLLM.
