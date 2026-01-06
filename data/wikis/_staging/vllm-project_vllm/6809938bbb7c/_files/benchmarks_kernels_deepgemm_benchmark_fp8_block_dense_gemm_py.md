# File: `benchmarks/kernels/deepgemm/benchmark_fp8_block_dense_gemm.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 435 |
| Functions | `benchmark_shape`, `format_table_row`, `print_table`, `format_speedup`, `run_benchmarks` |
| Imports | time, torch, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarks DeepGEMM FP8 block-wise dense matrix multiplication kernels.

**Mechanism:** Tests DeepGEMM's optimized FP8 GEMM implementation with block quantization, comparing against standard CUTLASS and cuBLAS implementations.

**Significance:** DeepGEMM aims to provide highly-optimized FP8 kernels. Benchmarking validates performance claims and identifies use cases where it excels.
