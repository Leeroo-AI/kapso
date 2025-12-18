# File: `benchmarks/benchmark_ngram_proposer.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 215 |
| Functions | `benchmark_propose`, `benchmark_batched_propose`, `invoke_main` |
| Imports | benchmark_utils, gc, numpy, tabulate, time, unittest, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarks N-gram speculative decode drafting performance.

**Mechanism:** Tests two modes: (1) Non-batched mode measures individual NgramProposer.propose() calls with varying max n-gram sizes (default: 5, 7, 10, 15, 20) over multiple iterations, reporting average and max times in microseconds. (2) Batched mode benchmarks the full batched proposal pipeline including batch preparation overhead for realistic workloads. Uses dummy input batches with configurable request counts and token lengths. Supports configurable min/max n-gram ranges and number of speculative tokens.

**Significance:** Performance validation tool for the v1 architecture's n-gram-based speculative decoding implementation. N-gram speculation can significantly improve throughput by predicting multiple tokens ahead without additional model calls. Critical for understanding the latency overhead of different n-gram window sizes and optimizing the speculative decoding configuration. Helps balance speculation benefit vs. proposal overhead.
