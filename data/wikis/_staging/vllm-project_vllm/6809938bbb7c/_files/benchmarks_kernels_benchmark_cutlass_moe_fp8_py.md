# File: `benchmarks/kernels/benchmark_cutlass_moe_fp8.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 406 |
| Functions | `bench_run`, `main` |
| Imports | nvtx, torch, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarks CUTLASS FP8 MoE kernels against Triton FP8 fused MoE kernels, comparing performance with FP8 quantized weights and 16-bit activations across various batch sizes and quantization strategies.

**Mechanism:** Creates test data with FP8 quantized expert weights and activations, runs both CUTLASS and Triton MoE implementations with CUDA graphs for reliable performance measurement, and compares execution times. Supports per-tensor and per-channel quantization modes with configurable batch sizes and model shapes.

**Significance:** Critical for evaluating the CUTLASS MoE FP8 kernel implementation against the existing Triton solution. Helps identify performance regressions and validate optimization decisions for different MoE model configurations (Mixtral, DeepSeek-V2, etc.).
