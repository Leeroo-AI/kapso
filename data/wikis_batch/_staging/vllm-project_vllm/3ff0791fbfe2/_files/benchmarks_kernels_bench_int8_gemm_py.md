# File: `benchmarks/kernels/bench_int8_gemm.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 169 |
| Functions | `build_int8_runner`, `benchmark`, `prepare_shapes` |
| Imports | argparse, copy, itertools, torch, vllm, weight_shapes |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmark INT8 GEMM variants

**Mechanism:** Similar to FP8 benchmark but for INT8 quantization. Tests tensor/channel weight quantization, tensor/token activation quantization, with/without dynamic quantization. Uses real model shapes across TP sizes and batch sizes (1-16K). Measures TFLOP/s with Triton benchmarking framework.

**Significance:** INT8 quantization performance validation. Provides performance data for INT8 as alternative to FP8, particularly important for hardware without native FP8 support. Helps users understand INT8 vs FP8 trade-offs.
