# File: `benchmarks/kernels/benchmark_quant.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 109 |
| Functions | `main` |
| Imports | time, torch, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarks scaled quantization operations (INT8 and FP8) with static or dynamic scaling.

**Mechanism:** Tests scaled_int8_quant and scaled_fp8_quant kernels across different tensor sizes, measuring kernel execution time with optional profiling support.

**Significance:** Essential for evaluating quantization kernel performance which is critical for quantized inference latency.
