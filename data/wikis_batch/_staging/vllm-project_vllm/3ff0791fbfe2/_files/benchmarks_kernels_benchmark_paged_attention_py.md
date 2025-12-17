# File: `benchmarks/kernels/benchmark_paged_attention.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 250 |
| Functions | `main` |
| Imports | random, time, torch, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarks legacy paged attention kernels

**Mechanism:** Tests paged_attention_v1, paged_attention_v2, and paged_attention_rocm (custom ROCm) implementations. Creates random KV cache blocks (128K blocks), block tables for sequences, and query tensors. Supports various head sizes (64-256), block sizes (16/32), and KV cache dtypes (auto/fp8). For v2, allocates partitioned outputs (PARTITION_SIZE=512 for CUDA, 256/1024 for ROCm). Measures kernel execution time using CUDA events. Includes optional CUDA profiler integration. Note: Script warns this is no longer the default path in vLLM inference.

**Significance:** Legacy benchmark for paged attention development and validation. While vLLM has moved to FlashAttention and other optimized attention mechanisms, this benchmark remains useful for: (1) validating paged attention correctness, (2) comparing against newer implementations, (3) ROCm-specific attention optimization. Historical significance for paged attention development in vLLM.
