# File: `benchmarks/kernels/bench_fp8_gemm.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 159 |
| Functions | `build_fp8_runner`, `benchmark`, `prepare_shapes` |
| Imports | argparse, copy, itertools, torch, vllm, weight_shapes |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmark FP8 GEMM variants

**Mechanism:** Compares BF16 vs FP8 quantized GEMM with various configurations: weight quantization (tensor/channel), activation quantization (tensor/token), with/without dynamic quantization. Tests real model shapes (Llama-3.1-8B default) across TP sizes and batch sizes (1-16K). Measures TFLOP/s using Triton benchmarking.

**Significance:** Comprehensive FP8 quantization performance analysis. Demonstrates trade-offs between quantization granularity and speed, helping users choose optimal FP8 configuration for their models. Essential for FP8 adoption decisions.
