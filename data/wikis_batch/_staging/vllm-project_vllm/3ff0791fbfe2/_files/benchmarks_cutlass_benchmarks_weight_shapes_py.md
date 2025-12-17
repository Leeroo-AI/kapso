# File: `benchmarks/cutlass_benchmarks/weight_shapes.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 46 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Reference model GEMM shapes

**Mechanism:** Defines WEIGHT_SHAPES dictionary mapping popular LLM models (Mistral-7B, Llama-2/3 variants, Llama-70B) to their layer dimensions [K, N] with tensor parallelism split dimensions. Each entry specifies QKV projection and MLP layer shapes for TP1 configurations.

**Significance:** Shared reference data for realistic benchmark shapes. Ensures all CUTLASS benchmarks test against actual production model dimensions, enabling meaningful performance predictions for real deployments.
