# File: `benchmarks/benchmark_batch_invariance.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 380 |
| Functions | `run_benchmark_with_batch_invariant`, `main` |
| Imports | contextlib, os, random, time, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Measure VLLM_BATCH_INVARIANT mode overhead

**Mechanism:** Runs identical workloads with VLLM_BATCH_INVARIANT disabled (baseline) and enabled, comparing initialization time, average trial time, and throughput. Uses randomized batch sizes and a needle prompt to verify output consistency. Generates random prompts with configurable lengths and runs multiple trials for statistical significance. Requires CUDA with SM90+ (Hopper architecture).

**Significance:** Performance validation tool for batch-invariant execution mode, which ensures deterministic outputs regardless of batch composition. Helps quantify the performance cost of enabling this consistency feature for applications requiring reproducible results.
