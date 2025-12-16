# File: `unsloth/kernels/moe/benchmark/utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 228 |
| Functions | `create_merged_results`, `post_process_results`, `save_results`, `create_kernel_configs`, `power_of_two_range`, `multiples_of_range`, `map_key_to_args`, `save_autotune_results`, `get_autotuner`, `postprocess_autotune_results` |
| Imports | argparse, datetime, grouped_gemm, itertools, json, logging, math, os, pandas, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides utility functions for MoE kernel benchmarking, including configuration generation, results processing, and autotuning cache management.

**Mechanism:**
- Generates kernel configuration combinations (block sizes, warps, stages, TMA settings) using power-of-two ranges
- Prunes invalid configurations (e.g., permute_x with TMA load x)
- Converts benchmark results to pandas DataFrames for analysis
- Saves results to CSV files with timestamps
- Manages autotuning cache by extracting and saving best configurations to JSON
- Retrieves autotuner instances for forward, dW, and dX kernels

**Significance:** Essential infrastructure for systematic kernel performance evaluation. Enables reproducible benchmarking by standardizing configuration generation and result storage. The autotuning cache management helps preserve optimal kernel parameters for different hardware configurations.