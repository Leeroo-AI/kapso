# File: `benchmarks/kernels/benchmark_cutlass_fp4_moe.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 504 |
| Functions | `to_fp8`, `bench_run`, `main` |
| Imports | nvtx, torch, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmark NVFP4 MOE vs FP8 MOE

**Mechanism:** Compares CUTLASS FP4 MOE (block-scaled, 16-element blocks) against Triton FP8 MOE (tensor-scaled) for DeepSeek-R1-FP4 model shapes (256 experts, topk=8). Tests both raw kernel calls and CUDA graph execution across batch sizes (4-2048). Uses NVTX annotations for profiling. Configures VllmConfig for proper pipeline parallelism.

**Significance:** Critical evaluation for MoE model quantization. FP4 block quantization enables extreme compression for massive expert counts while maintaining quality. Essential for validating DeepSeek-R1 and similar large-scale MoE deployment on Blackwell hardware with 4-bit experts.
