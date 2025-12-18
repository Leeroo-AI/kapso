# File: `benchmarks/kernels/benchmark_rmsnorm.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 255 |
| Classes | `HuggingFaceRMSNorm` |
| Functions | `rmsnorm_naive`, `rmsnorm_flashinfer`, `rmsnorm_vllm`, `calculate_diff`, `get_benchmark` |
| Imports | flashinfer, itertools, torch, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarks RMSNorm kernel with optional residual connections across different data types.

**Mechanism:** Creates test tensors and RMSNorm layers, runs timed benchmarks with optional CUDA profiling support.

**Significance:** RMSNorm is used extensively in modern LLMs (LLaMA, etc.). Critical for overall model performance.
