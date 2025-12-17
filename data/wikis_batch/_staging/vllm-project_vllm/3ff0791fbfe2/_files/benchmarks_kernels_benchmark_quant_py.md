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

**Purpose:** Benchmarks FP8/INT8 quantization kernels

**Mechanism:** Simple benchmark for scaled_int8_quant and scaled_fp8_quant operations. Creates random tensors of specified shape (num_tokens x hidden_size) and optional static scale. Measures execution time with warmup iterations and CUDA synchronization. Supports profiler integration for detailed kernel analysis. Tests both static scale (pre-computed) and dynamic scale (computed per invocation) modes.

**Significance:** Basic validation tool for quantization kernel performance. Essential for verifying that quantization operations (converting FP16/BF16 to INT8/FP8) meet latency requirements. Used to measure overhead of dynamic quantization vs static quantization. Important for understanding quantization costs in inference pipelines.
