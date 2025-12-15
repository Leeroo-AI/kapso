# File: `unsloth/kernels/moe/benchmark/utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 228 |
| Functions | `create_merged_results`, `post_process_results`, `save_results`, `create_kernel_configs`, `power_of_two_range`, `multiples_of_range`, `map_key_to_args`, `save_autotune_results`, `... +2 more` |
| Imports | argparse, datetime, grouped_gemm, itertools, json, logging, math, os, pandas, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Utility functions for MoE benchmarking and kernel configuration management

**Mechanism:** Provides functions to create kernel configurations from parameter ranges, process/save benchmark results as CSV/JSON, retrieve autotuners for different modes (forward, dX, dW), and manage autotuning cache. Handles result formatting and persistence.

**Significance:** Essential infrastructure for systematic performance testing. Enables reproducible benchmarking by managing kernel configurations, saving results, and extracting best-performing configurations from autotuning runs.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
