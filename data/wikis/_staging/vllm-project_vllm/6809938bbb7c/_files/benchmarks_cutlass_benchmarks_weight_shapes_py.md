# File: `benchmarks/cutlass_benchmarks/weight_shapes.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 46 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines realistic weight matrix shapes for benchmarking popular LLM models.

**Mechanism:** WEIGHT_SHAPES dictionary maps model names (Mistral-7B, Llama-2-7b/13b/70b, Llama-3-8b) to lists of [K, N] weight dimensions with tensor-parallel split dimensions. Format is ([K, N], TP_SPLIT_DIM) where TP_SPLIT_DIM indicates which dimension is sharded across tensor-parallel ranks (0 for K-dimension, 1 for N-dimension). Includes attention projection matrices (QKV, O) and MLP matrices (gate/up, down) for each model architecture.

**Significance:** Reference data for realistic GEMM benchmarking. Using actual model weight shapes ensures benchmark results reflect production workload characteristics. The TP_SPLIT_DIM annotations enable testing how tensor parallelism affects GEMM performance across different sharding strategies. Critical for validating that kernel optimizations work well on real model architectures, not just synthetic square matrices. Shared across multiple benchmark scripts for consistency.
