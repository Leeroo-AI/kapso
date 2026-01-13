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

**Purpose:** Benchmarking script that compares performance of Triton grouped GEMM MoE implementations against HuggingFace reference models (Llama4 and Qwen3).

**Mechanism:** Provides `run_benchmark_forward` and `run_benchmark_backward` functions that measure timing using Triton's `do_bench`. The `setup_model` function instantiates both reference (HF) and Triton-optimized MoE blocks. Supports autotuning mode with configurable kernel parameters (block sizes, warps, stages, TMA options). Uses argparse for CLI configuration including model selection, sequence length, dtype, and permutation options.

**Significance:** Essential development tool for validating performance gains of custom Triton MoE kernels over standard implementations, enabling kernel optimization and regression testing.
