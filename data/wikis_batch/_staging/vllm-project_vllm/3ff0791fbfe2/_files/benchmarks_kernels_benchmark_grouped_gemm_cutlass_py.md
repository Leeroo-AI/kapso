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

**Purpose:** Benchmarks CUTLASS grouped GEMM for FP8 MoE

**Mechanism:** Compares Triton MoE and CUTLASS grouped GEMM implementations for FP8 quantized MoE layers. Creates FP8 quantized weights and scales, benchmarks both with and without CUDA graphs across multiple model configurations (Mixtral, DeepSeek-V2-Lite, Granite) and batch sizes (1-512). Uses torch.utils.benchmark to measure execution time, captures 10 operations per graph for amortized timing.

**Significance:** Validates CUTLASS grouped GEMM performance for MoE workloads. Part of the evaluation suite for selecting optimal MoE kernel implementation. Helps determine when to use CUTLASS vs Triton for FP8 quantized mixture-of-experts inference.
