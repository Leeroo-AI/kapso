# File: `unsloth/kernels/moe/benchmark/utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 228 |
| Functions | `create_merged_results`, `post_process_results`, `save_results`, `create_kernel_configs`, `power_of_two_range`, `multiples_of_range`, `map_key_to_args`, `save_autotune_results`, `... +2 more` |
| Imports | argparse, datetime, grouped_gemm, itertools, json, logging, math, os, pandas, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Utility functions for benchmark configuration, kernel config generation, result processing and storage.

**Mechanism:** Creates kernel configs from parameter ranges, prunes invalid configs, processes results into DataFrames, manages autotuning cache, serializes results to CSV/JSON.

**Significance:** Supports benchmarking infrastructure and results analysis for MoE kernel optimization.
