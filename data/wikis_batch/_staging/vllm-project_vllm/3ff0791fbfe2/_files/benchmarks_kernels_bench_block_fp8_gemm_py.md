# File: `benchmarks/kernels/bench_block_fp8_gemm.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 160 |
| Functions | `build_w8a8_block_fp8_runner`, `benchmark_tflops` |
| Imports | os, torch, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmark W8A8 block FP8 GEMM

**Mechanism:** Tests block-quantized FP8 matrix multiplication (DeepSeek-V3 shapes) with 128x128 block size. Compares BF16 baseline against Triton and CUTLASS implementations. Measures TFLOP/s across batch sizes (1-16K). Disables DeepGEMM to isolate CUTLASS performance. Uses Triton's perf_report decorator.

**Significance:** Validation tool for block-wise FP8 quantization performance. Block quantization provides finer granularity than per-tensor while maintaining efficiency. Essential for DeepSeek-V3 and similar models using grouped FP8 quantization schemes.
