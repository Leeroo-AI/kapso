# File: `benchmarks/kernels/bench_nvfp4_gemm.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 198 |
| Functions | `build_nvfp4_runner`, `benchmark`, `prepare_shapes` |
| Imports | argparse, copy, itertools, os, torch, vllm, weight_shapes |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmark NVFP4 quantized GEMM

**Mechanism:** Tests NVIDIA's FP4 format (requires Blackwell/SM100+) using CUTLASS or FBGEMM implementations. Compares BF16 baseline against NVFP4 with/without activation quantization. Uses global scaling (FP8_E4M3 * FP4_E2M1) for quantization. Tests Llama-3.1-8B shapes across batch sizes (1-16K) and TP sizes.

**Significance:** Blackwell architecture evaluation tool. NVFP4 enables 4-bit inference with hardware acceleration on latest GPUs. Essential for understanding next-gen GPU quantization capabilities and comparing against software-based 4-bit solutions.
