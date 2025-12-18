# File: `benchmarks/kernels/benchmark_machete.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 745 |
| Classes | `BenchmarkTensors`, `TypeConfig`, `ToTorchDtype` |
| Functions | `terse_type_name`, `rand_data`, `quantize_and_pack`, `create_bench_tensors`, `torch_matmul_f16_create_bench_fn`, `cutlass_scaled_mm_create_bench_fn`, `marlin_create_bench_fn`, `machete_create_bench_fn`, `... +9 more` |
| Imports | argparse, collections, copy, dataclasses, itertools, math, os, pandas, pickle, time, ... +3 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarks Machete mixed-precision GEMM kernels against torch.matmul, Marlin, and CUTLASS implementations for weight-quantized matrix multiplications with various quantization schemes.

**Mechanism:** Creates quantized weight matrices with group scales, channel scales, and token scales. Benchmarks Machete kernel with different schedules, comparing against torch.matmul baseline, Marlin quantized GEMM, and CUTLASS scaled_mm. Supports schedule sweeping to find optimal kernel configuration for given problem sizes.

**Significance:** Critical for evaluating Machete's mixed-precision GEMM performance which enables efficient INT4/FP8 weight quantization with flexible scaling strategies. Essential for optimizing quantized inference across different model architectures and batch sizes.
