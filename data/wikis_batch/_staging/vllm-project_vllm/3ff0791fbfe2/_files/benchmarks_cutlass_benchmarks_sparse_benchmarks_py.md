# File: `benchmarks/cutlass_benchmarks/sparse_benchmarks.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 515 |
| Functions | `bench_fn`, `bench_int8`, `bench_fp8`, `bench`, `print_timers`, `run`, `make_output`, `run_square_bench`, `... +2 more` |
| Imports | argparse, collections, copy, itertools, pickle, time, torch, utils, vllm, weight_shapes |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmark 2:4 sparse GEMM

**Mechanism:** Comprehensive benchmarking of 2:4 structured sparse matrix multiplication using CUTLASS kernels. Tests INT8/FP8 sparse operations with various quantization schemes (per-tensor, per-token, per-channel, per-group). Includes baseline dense comparisons, supports real model shapes (Llama, Mistral) with tensor parallelism, and measures TFLOP/s. Generates detailed comparison tables and pickled results.

**Significance:** Critical for validating 2:4 sparsity performance gains in quantized models. 2:4 structured sparsity (50% zero weights) enables 2x speedup on Ampere+ GPUs while maintaining model quality. Essential for proving sparse quantized inference viability.
