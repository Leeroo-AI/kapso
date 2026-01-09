# File: `unsloth/kernels/moe/benchmark/benchmark_fused_moe.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 399 |
| Functions | `run_benchmark_forward`, `run_benchmark_backward`, `setup_model`, `run_benchmark` |
| Imports | argparse, contextlib, grouped_gemm, time, torch, transformers, triton, utils |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive benchmarking script for evaluating MoE kernel performance across Llama4 and Qwen3 architectures.

**Mechanism:** Compares reference HuggingFace implementations against Triton-optimized grouped GEMM kernels in forward, backward (dW), and backward (dX) modes. Supports autotuning across block sizes, warp counts, stages, and TMA configurations. Uses triton.testing.do_bench for accurate timing and saves results with metadata for analysis.

**Significance:** Critical for performance validation and optimization. Ensures Unsloth's fused MoE kernels achieve speedups over standard implementations. The autotuning results guide production kernel configurations for different model sizes and hardware architectures.
