# File: `benchmarks/kernels/benchmark_grouped_gemm_cutlass.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 427 |
| Functions | `to_fp8`, `bench_run`, `main` |
| Imports | benchmark_shapes, torch, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarks CUTLASS grouped GEMM MoE implementation with FP8 quantization against Triton FP8 fused MoE across various model configurations and batch sizes.

**Mechanism:** Tests both Triton and CUTLASS MoE kernels with and without CUDA graphs. Creates FP8 quantized weights and activations, runs benchmarks for different MKN dimensions and expert configurations, and compares performance metrics.

**Significance:** Validates the CUTLASS grouped GEMM approach for MoE layers as an alternative to Triton kernels. Important for understanding trade-offs between different GEMM implementations for mixture-of-experts models with varying batch sizes and expert counts.
