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

**Purpose:** Benchmarks fused MoE (Mixture of Experts) implementations for Llama4 and Qwen3 models, comparing reference implementations against optimized grouped GEMM kernels.

**Mechanism:**
- Sets up reference HuggingFace models and custom Triton-optimized versions
- Runs forward and backward passes with configurable kernel parameters
- Measures execution time using triton.testing.do_bench
- Supports autotuning or manual kernel configuration (block sizes, warps, stages, TMA loads)
- Can benchmark forward pass, backward dW, backward dX, or full backward separately
- Saves autotuning results and performance metrics to CSV files

**Significance:** Critical performance validation tool for the MoE kernel optimizations. Provides empirical evidence of speedups achieved by the fused grouped GEMM implementation over standard PyTorch operations. Used for kernel parameter tuning and performance regression testing.