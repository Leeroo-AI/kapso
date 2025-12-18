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

**Purpose:** Benchmarks per-token group quantization operations for dynamic FP8 quantization.

**Mechanism:** Measures performance of quantizing activations with per-token or per-group granularity, critical for FP8 inference pipelines.

**Significance:** Important for evaluating dynamic quantization overhead which is a key component of FP8 inference efficiency.
