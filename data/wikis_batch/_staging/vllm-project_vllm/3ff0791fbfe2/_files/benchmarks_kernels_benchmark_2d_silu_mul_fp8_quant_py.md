# File: `benchmarks/kernels/benchmark_2d_silu_mul_fp8_quant.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 244 |
| Classes | `ImplType`, `BenchmarkTensors` |
| Functions | `print_timers`, `reference_quant`, `reference`, `bench_impl`, `test_correctness`, `run` |
| Imports | dataclasses, enum, itertools, torch, typing, utils, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmark fused SiLU-mul-quant kernel

**Mechanism:** Compares fused (silu_mul_per_token_group_quant_fp8_colmajor) vs unfused (silu_and_mul + quant) implementations. Tests across token counts (128-2048 & 2K-128K) and hidden dimensions (2K-8K) with group size 128. Uses CUDA graphs with argument pools (size 8) for realistic measurements. Validates correctness before benchmarking.

**Significance:** Performance validation for MLP activation fusion. SiLU-multiply-quantize pattern is critical in LLM feedforward layers. Fusing these operations reduces memory traffic and improves latency, essential optimization for quantized inference throughput.
