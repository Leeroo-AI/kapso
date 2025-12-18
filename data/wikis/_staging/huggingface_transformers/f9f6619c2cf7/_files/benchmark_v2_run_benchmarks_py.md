# File: `benchmark_v2/run_benchmarks.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 128 |
| Imports | argparse, framework, json, logging, sys, uuid |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Next-generation benchmark orchestration script with configurable coverage levels and Hub integration. Provides systematic benchmarking across different attention implementations, compile modes, and parameter combinations.

**Mechanism:** Accepts multiple configuration sources (coverage levels 0-4 or JSON/JSONL config files) and adapts them to specified batch sizes, sequence lengths, and token generation counts. Uses BenchmarkRunner framework to execute benchmarks with optional GPU monitoring and profiling. Supports multi-dimensional parameter sweeps (batch size, sequence length, tokens to generate) and can push results directly to HuggingFace Hub datasets. Validates inputs (e.g., num_tokens_to_generate > 1 for ITL computation).

**Significance:** Represents the v2 evolution of transformers benchmarking with more sophisticated configuration management and framework integration. The level-based coverage system (0-4) allows scaling from quick smoke tests to exhaustive cross-product benchmarks including all attention implementations and compilation modes. Essential for comprehensive performance evaluation and automated regression detection in CI/CD pipelines.
