# File: `benchmarks/cutlass_benchmarks/w8a8_benchmarks.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 372 |
| Functions | `bench_fn`, `bench_int8`, `bench_fp8`, `bench`, `print_timers`, `run`, `make_output`, `run_square_bench`, `... +2 more` |
| Imports | argparse, collections, copy, itertools, pickle, time, torch, utils, vllm, weight_shapes |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarks W8A8 (8-bit weight, 8-bit activation) quantized GEMM kernels for INT8 and FP8.

**Mechanism:** Compares multiple implementations: PyTorch dense matmul (BF16/FP16 baseline), PyTorch _scaled_mm (FP8 with/without fast_accum), CUTLASS scaled_mm (INT8/FP8 with optional bias), CUTLASS scaled_mm_azp (INT8 with activation zero-point), and Triton block-scaled matmul. Tests both per-tensor and block-wise (128x128) quantization scales. Runs three benchmark modes: square_bench, range_bench, and model_bench using realistic shapes from LLaMA/Mistral models. Supports selective kernel benchmarking via --kernels flag. Pickles results for analysis.

**Significance:** Critical performance validation for quantized inference. W8A8 quantization reduces memory bandwidth and enables faster compute on tensor cores. Compares CUTLASS custom kernels against PyTorch's native FP8 support to validate optimization benefits. Block-wise scaling (vs. per-tensor) improves accuracy at the cost of additional overhead - benchmarks quantify this tradeoff. Essential for selecting optimal quantization strategy for production serving.
