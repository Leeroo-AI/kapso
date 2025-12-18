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

**Purpose:** Provides shared utility classes and functions for benchmark scripts.

**Mechanism:** TimeCollector implements a context manager for measuring operation durations with nanosecond precision, supporting different time scales (NS, US, MS, S). Tracks count, sum, and maximum times to compute averages and maxes. InfEncoder is a JSON encoder that handles infinity values by converting them to the string "inf". convert_to_pytorch_benchmark_format() transforms results into PyTorch OSS benchmark database format with metadata. write_to_json() persists benchmark data with custom encoding for non-serializable objects.

**Significance:** Core infrastructure for benchmark consistency. TimeCollector is used across multiple benchmarks (block_pool, ngram_proposer, etc.) to standardize timing measurements. The PyTorch format converter enables integration with broader ML benchmarking ecosystems. Ensures reproducible and comparable benchmark results across different vLLM components and versions.
