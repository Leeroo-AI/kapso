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

**Purpose:** Benchmarks Triton grouped GEMM kernels against reference implementations for Llama4 and Qwen3 MoE models.

**Mechanism:** Implements forward/backward benchmark harness, supports manual kernel config or autotuning, compares PyTorch vs Triton implementations, measures speedup metrics with proper warmup and timing.

**Significance:** Critical for performance validation and kernel optimization - drives decision-making on kernel parameters and validates correctness.
