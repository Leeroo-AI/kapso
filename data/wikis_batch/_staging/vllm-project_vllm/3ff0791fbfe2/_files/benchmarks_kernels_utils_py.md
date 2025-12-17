# File: `benchmarks/kernels/utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 214 |
| Classes | `CudaGraphBenchParams`, `ArgPool`, `Bench`, `ArgsIterator` |
| Imports | collections, dataclasses, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmark utility classes and helpers

**Mechanism:** Provides infrastructure for kernel benchmarking: CudaGraphBenchParams for CUDA graph configuration, ArgPool for rotating through argument values across iterations, Bench class with ArgsIterator for managing benchmark parameters. Includes timer wrappers and measurement utilities building on torch.utils.benchmark. Supports parametric benchmarking where arguments cycle through value pools to prevent cache effects.

**Significance:** Shared utility framework for consistent benchmarking across vLLM. Provides abstractions that simplify writing new benchmarks while ensuring accurate measurements. ArgPool pattern prevents artificial speedups from cache hits. Essential infrastructure for maintainable benchmark suite. Reduces code duplication across benchmark scripts.
