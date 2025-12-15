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

**Purpose:** Benchmarks fused MoE implementations against reference implementations

**Mechanism:** Compares forward/backward pass performance of Triton grouped GEMM kernels against HuggingFace reference models (Llama4, Qwen3). Tests various configurations including permutation strategies, autotuning, and kernel parameters. Saves benchmark results and autotuning cache for analysis.

**Significance:** Critical performance validation tool that ensures the custom Triton kernels provide speedups while maintaining numerical accuracy. Used to optimize kernel configurations for different model architectures.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
