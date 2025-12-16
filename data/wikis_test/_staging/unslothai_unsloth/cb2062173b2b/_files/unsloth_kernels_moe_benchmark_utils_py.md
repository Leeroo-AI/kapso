# File: `unsloth/kernels/moe/benchmark/utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 228 |
| Functions | `create_merged_results`, `post_process_results`, `save_results`, `create_kernel_configs`, `power_of_two_range`, `multiples_of_range`, `map_key_to_args`, `save_autotune_results`, `... +2 more` |
| Imports | argparse, datetime, grouped_gemm, itertools, json, logging, math, os, pandas, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Utility functions for benchmark configuration, result processing, and autotuning support for MoE kernel benchmarks.

**Mechanism:**
- `create_kernel_configs()`: Generates Cartesian product of kernel configurations from parameter ranges (block sizes, warps, stages, TMA options)
- `power_of_two_range()` / `multiples_of_range()`: Generate parameter ranges for kernel tuning
- `post_process_results()` / `save_results()`: Convert KernelResult objects to pandas DataFrames and save to CSV with timestamps
- `get_autotuner()`: Returns appropriate autotuned kernel function for forward/dW/dX/backward modes
- `save_autotune_results()`: Persists autotuned configurations to JSON files organized by device and timestamp
- `postprocess_autotune_results()`: Extracts and saves best kernel configurations from Triton's autotuner cache

**Significance:** Essential infrastructure for systematic kernel performance optimization. Enables grid search over kernel parameters, automated result tracking, and preservation of optimal configurations for reuse. Supports the benchmarking workflow by handling configuration management and data persistence.
