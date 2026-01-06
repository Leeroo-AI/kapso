# File: `benchmarks/kernels/weight_shapes.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 104 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines weight matrix shapes with tensor-parallel split dimensions for benchmarking various models.

**Mechanism:** Maps model names to lists of ([K, N], TP_SPLIT_DIM) tuples indicating layer dimensions and which dimension gets split under tensor parallelism.

**Significance:** Centralized shape definitions ensure benchmarks test realistic configurations. More structured than benchmark_shapes.py with explicit TP split information.
