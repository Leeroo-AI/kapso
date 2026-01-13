# File: `unsloth/kernels/moe/benchmark/utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 228 |
| Functions | `create_merged_results`, `post_process_results`, `save_results`, `create_kernel_configs`, `power_of_two_range`, `multiples_of_range`, `map_key_to_args`, `save_autotune_results`, `... +2 more` |
| Imports | argparse, datetime, grouped_gemm, itertools, json, logging, math, os, pandas, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Utility functions for MoE kernel benchmarking, including result processing, kernel configuration generation, and autotuning result management.

**Mechanism:** Provides `create_kernel_configs` to generate parameter sweeps over block sizes, warps, and stages using Cartesian products. Includes `power_of_two_range` and `multiples_of_range` helpers for parameter range generation. The `post_process_results` and `save_results` functions create pandas DataFrames and save to CSV. The `get_autotuner` function retrieves the appropriate autotuned kernel based on mode (forward/backward/dW/dX). Configuration pruning removes invalid TMA + permute combinations.

**Significance:** Supporting utility module that enables systematic kernel tuning experiments and performance tracking for the MoE grouped GEMM benchmarking workflow.
