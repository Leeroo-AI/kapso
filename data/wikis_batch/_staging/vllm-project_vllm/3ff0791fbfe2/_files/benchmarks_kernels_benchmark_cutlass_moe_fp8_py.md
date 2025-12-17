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

**Purpose:** Benchmarks CUTLASS FP8 MoE vs Triton FP8 MoE kernels

**Mechanism:** Compares two FP8-quantized MoE implementations using CUDA graphs. Creates FP8 quantized weights and activations, then benchmarks both cutlass_moe_fp8 and triton fused_experts kernels across different model configurations (Mixtral, DeepSeek-V2, GLM-4, etc.) and batch sizes. Measures kernel execution time using CUDA events and reports performance in microseconds.

**Significance:** Performance comparison tool for selecting optimal FP8 MoE kernel implementation. Validates that CUTLASS implementation can compete with or outperform Triton implementation for mixture-of-experts models with FP8 quantization. Essential for quantized MoE inference optimization.
