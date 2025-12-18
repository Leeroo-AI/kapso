# File: `benchmarks/kernels/utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 214 |
| Classes | `CudaGraphBenchParams`, `ArgPool`, `Bench`, `ArgsIterator` |
| Imports | collections, dataclasses, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Shared utility classes and functions for kernel benchmarking infrastructure.

**Mechanism:** Provides CudaGraphBenchParams for CUDA graph benchmarking, ArgPool for parameter iteration, and Bench class for standardized benchmark execution with CUDA graph support.

**Significance:** Common infrastructure reduces code duplication and ensures consistent benchmarking methodology across different kernel benchmarks.
