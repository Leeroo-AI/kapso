# File: `benchmarks/kernels/benchmark_layernorm.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 94 |
| Functions | `main` |
| Imports | time, torch, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarks RMSNorm kernel performance

**Mechanism:** Simple benchmark for RMSNorm layer with optional residual connection. Creates random tensors of specified shape (num_tokens x hidden_size), runs RMSNorm forward pass repeatedly, measures execution time using CUDA synchronization. Supports different dtypes (half/bfloat16/float) and optional CUDA profiler integration for detailed kernel analysis.

**Significance:** Basic performance validation tool for RMSNorm implementation. Used to verify kernel optimization effectiveness and measure latency for different tensor sizes and dtypes. Part of the kernel optimization validation suite for transformer layer normalization.
