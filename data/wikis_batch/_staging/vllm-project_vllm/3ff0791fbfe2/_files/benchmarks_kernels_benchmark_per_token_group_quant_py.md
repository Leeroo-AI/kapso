# File: `benchmarks/kernels/benchmark_per_token_group_quant.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 159 |
| Functions | `parse_args` |
| Imports | argparse, collections, contextlib, math, torch, unittest, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Compares CUDA vs Triton per-token group quantization

**Mechanism:** Benchmarks per_token_group_quant for FP8 and INT8 quantization, comparing CUDA implementation against Triton fallback. Uses mock patching to force Triton path. Tests various configurations: shapes ((32,128), (64,256), (16,512)), group sizes (64, 128), FP8 options (column_major scales, UE8M0 encoding), and INT8 variants. Measures execution time with warmup and benchmark iterations using CUDA events. Reports speedup (Triton time / CUDA time) for each configuration.

**Significance:** Validates CUDA kernel performance advantage over Triton for per-token group quantization. This operation is critical for activation quantization in FP8/INT8 inference, computing scales per group within each token. Helps ensure CUDA path is preferred when available. Important for quantized inference performance, especially for models using per-group quantization schemes.
