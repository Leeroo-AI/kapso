# File: `benchmarks/kernels/benchmark_shapes.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 94 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines weight matrix shapes for various models across different tensor-parallel configurations used by other benchmark scripts.

**Mechanism:** Contains WEIGHT_SHAPES and WEIGHT_SHAPES_MOE dictionaries mapping model names to lists of [K, N] dimensions and TP split dimensions for linear layers and MoE expert layers.

**Significance:** Central configuration file that allows benchmarks to test realistic model layer shapes. Essential for ensuring benchmarks reflect actual production workloads.
