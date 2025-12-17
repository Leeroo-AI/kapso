# File: `benchmarks/kernels/benchmark_shapes.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 94 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines weight matrix shapes for benchmarks

**Mechanism:** Static dictionary WEIGHT_SHAPES containing realistic GEMM shapes from popular models (Mistral-7B, Llama-2-7b/13b/70b, Llama-3-8b) across different tensor parallelism sizes (TP1/TP2/TP4). Each model has 4 weight shapes representing: QKV projection, output projection, gate_up projection, and down projection. Includes "ideal" synthetic shape for testing. Shapes specified as [K, N] dimensions for matrix multiplication benchmarking.

**Significance:** Provides standardized, realistic test cases for GEMM kernel benchmarks. Ensures benchmarks reflect actual model workloads rather than synthetic patterns. Critical reference for Marlin, Machete, and other quantized GEMM benchmarks. Helps validate kernel performance on production-relevant matrix sizes. Shared across multiple benchmark scripts for consistency.
