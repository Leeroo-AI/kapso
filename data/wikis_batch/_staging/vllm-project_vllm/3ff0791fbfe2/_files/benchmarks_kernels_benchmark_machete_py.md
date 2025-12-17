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

**Purpose:** Benchmarks Machete quantized GEMM kernel performance

**Mechanism:** Compares Machete (custom quantized GEMM) against torch.matmul, cutlass_scaled_mm, and Marlin across various quantization schemes. Supports multiple weight types (uint4b8, uint4), activation types (fp16/bf16/int8/fp8), group scales, zero points, channel scales, and token scales. Creates large weight sets exceeding L2 cache to ensure realistic measurements. Includes optional schedule sweeping to find optimal kernel configuration. Tests three modes: square_bench (square matrices), range_bench (custom ranges), and model_bench (from model weight shapes). Outputs timing results and saves as pickle files.

**Significance:** Key tool for evaluating Machete kernel competitiveness against established quantized GEMM implementations. Essential for validating performance across diverse quantization configurations and matrix shapes. Helps determine when Machete provides performance advantages over Marlin, CUTLASS, or other backends. Critical for quantized inference optimization strategy.
