# File: `benchmarks/benchmark_utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 125 |
| Classes | `InfEncoder`, `TimeCollector` |
| Functions | `convert_to_pytorch_benchmark_format`, `write_to_json` |
| Imports | argparse, json, math, os, time, types, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Shared utilities for benchmarks

**Mechanism:** Provides TimeCollector for microsecond-precision latency measurements (collect/report average/max), InfEncoder for JSON serialization handling infinity/NaN values, convert_to_pytorch_benchmark_format for standardizing result structures, and write_to_json for persistent result storage.

**Significance:** Core utility library for all vLLM benchmarks. Ensures consistent timing methodology and result formats across different benchmarking scripts. Essential for maintaining measurement accuracy and enabling cross-benchmark comparisons.
