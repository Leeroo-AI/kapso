# File: `benchmarks/kernels/benchmark_activation.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 105 |
| Functions | `benchmark_activation` |
| Imports | itertools, torch, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmark activation function kernels

**Mechanism:** Compares custom CUDA kernels vs torch.compile for various activation functions (silu_and_mul, gelu variants, fatrelu_and_mul, etc.). Tests across batch sizes (1-128), sequence lengths (1-4096), and intermediate dimensions (3K-12K). Uses CustomOp registry to instantiate layers with appropriate parameters (threshold for FATReLU, approximate for GELU).

**Significance:** Validation tool for activation kernel optimizations. Demonstrates performance benefits of specialized CUDA kernels over compiled PyTorch for common LLM activation patterns. Essential for choosing between custom vs compiled implementations.
