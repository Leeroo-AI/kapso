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

**Purpose:** Measures performance overhead of VLLM_BATCH_INVARIANT mode compared to baseline.

**Mechanism:** Runs the same workload twice - once with VLLM_BATCH_INVARIANT=0 (baseline) and once with VLLM_BATCH_INVARIANT=1 (batch invariant mode). Generates random prompts with a "needle" prompt mixed in at random positions within variable-sized batches. Measures initialization time, per-trial time, throughput (tokens/s), and compares the overhead percentage. Supports Hopper+ GPUs (SM90) and configurable model/batch size/temperature parameters.

**Significance:** Performance validation tool for the batch invariance feature, which ensures deterministic outputs regardless of batch composition. Helps quantify the computational cost of maintaining batch-independent execution. Critical for understanding the tradeoff between determinism and throughput. Targets DeepSeek-V3 and Qwen3 models with configurable tensor parallelism.
