# File: `benchmarks/kernels/benchmark_silu_mul_fp8_quant.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 720 |
| Functions | `silu_mul_fp8_quant_deep_gemm_triton`, `benchmark`, `create_comparison_plot`, `create_combined_plot`, `create_total_tokens_plot` |
| Imports | collections, matplotlib, numpy, torch, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive SiLU+Mul+FP8Quant benchmark suite

**Mechanism:** Extensive benchmark comparing CUDA (persistent_masked_m_silu_mul_quant) vs Triton (_silu_mul_fp8_quant_deep_gemm) implementations of fused SiLU activation, element-wise multiply, and FP8 quantization for MoE. Tests across multiple expert configurations, token distributions, hidden dimensions (4096/14336), and group sizes. Measures memory bandwidth utilization, calculates speedup ratios, and generates detailed performance plots (comparison, combined, total tokens). Simulates realistic MoE workloads with varying tokens-per-expert distributions. Supports UE8M0 scale encoding option.

**Significance:** Critical for MoE FP8 performance optimization. The fused SiLU+mul+quant operation is executed in MoE gate_up projection before expert GEMMs. Proper fusion and quantization are essential for efficient FP8 MoE inference. Benchmark validates that CUDA kernel provides superior performance over Triton for this operation. Important for DeepGEMM and FP8 MoE implementations. Directly impacts MoE inference throughput.
