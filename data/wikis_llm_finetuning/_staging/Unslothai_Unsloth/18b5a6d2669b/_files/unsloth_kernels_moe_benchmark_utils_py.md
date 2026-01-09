# File: `unsloth/kernels/moe/benchmark/utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 228 |
| Functions | `create_merged_results`, `post_process_results`, `save_results`, `create_kernel_configs`, `power_of_two_range`, `multiples_of_range`, `map_key_to_args`, `save_autotune_results`, `... +2 more` |
| Imports | argparse, datetime, grouped_gemm, itertools, json, logging, math, os, pandas, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Utility functions for benchmark result processing, kernel configuration generation, and autotuning result management.

**Mechanism:** Provides functions to create kernel configurations from parameter ranges (power-of-two and linear), merge benchmark results into dataframes, save results to CSV with timestamps, and extract/save autotuning cache data. Includes get_autotuner() to access the appropriate autotuned kernel for each mode (forward/dW/dX).

**Significance:** Essential infrastructure for the benchmarking pipeline. Standardizes result formats, enables reproducible experiments, and facilitates analysis of performance across different configurations. The autotuning utilities help identify optimal kernel parameters for production use.
