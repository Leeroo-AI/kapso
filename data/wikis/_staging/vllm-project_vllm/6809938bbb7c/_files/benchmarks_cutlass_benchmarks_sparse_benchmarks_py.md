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

**Purpose:** Benchmarks CUTLASS sparse (2:4 structured sparsity) GEMM kernels for INT8 and FP8 quantization.

**Mechanism:** Compares PyTorch dense matmul (BF16/FP16), CUTLASS dense scaled_mm, and CUTLASS sparse scaled_sparse_mm implementations. Generates 2:4 sparse tensors (2 non-zero elements per 4 elements) using prune_to_2_4(). Runs three benchmark modes: (1) square_bench for square matrices with varying dimensions, (2) range_bench for sweeping one dimension while keeping others constant, (3) model_bench using realistic weight shapes from LLaMA/Mistral models with tensor parallelism. Reports timing comparisons and pickles results for analysis.

**Significance:** Performance validation for sparse GEMM kernels supporting structured sparsity. 2:4 sparsity can reduce memory bandwidth and computation by ~50% with minimal accuracy loss. Critical for evaluating whether sparse kernels deliver expected speedups compared to dense operations. Helps determine if enabling structured sparsity improves throughput for quantized models (INT8/FP8) serving.
