# File: `benchmark_v2/run_benchmarks.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 128 |
| Imports | argparse, framework, json, logging, sys, uuid |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Top-level orchestrator for running performance benchmarks with configurable coverage levels and result tracking.

**Mechanism:** Parses extensive CLI arguments for benchmark parameters (batch size, sequence length, warmup/iterations, etc.), loads configs from file or generates them based on coverage level (0=single, 1=few, 2=attn variants, 3/4=full combinations), adapts configs to specified parameters, executes benchmarks via BenchmarkRunner with optional GPU monitoring and profiling, and optionally pushes results to a Hub dataset for tracking performance over time.

**Significance:** Central entry point for the benchmark_v2 framework that enables systematic performance testing across different model configurations, attention implementations, and compile modes. Supports both local development benchmarking and CI-integrated performance regression tracking with git metadata integration for historical analysis.
