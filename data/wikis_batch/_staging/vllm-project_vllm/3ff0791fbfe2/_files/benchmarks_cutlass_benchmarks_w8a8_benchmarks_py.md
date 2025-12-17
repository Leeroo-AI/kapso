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

**Purpose:** Benchmark W8A8 quantized GEMM

**Mechanism:** Comprehensive dense quantized matrix multiplication benchmarking (INT8 and FP8). Tests multiple quantization granularities (per-tensor/token/channel/group) with or without activation quantization. Supports real model shapes across TP sizes, measures TFLOP/s using PyTorch benchmark utilities, and generates detailed comparison tables.

**Significance:** Primary validation tool for INT8/FP8 quantized inference performance. Essential for demonstrating W8A8 quantization benefits (memory savings, throughput gains) against BF16 baseline across realistic LLM layer dimensions.
