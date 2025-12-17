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

**Purpose:** Benchmark ngram speculative decoding

**Mechanism:** Tests v1 NgramProposer performance for prompt lookup speculation. Runs two modes: single-request propose() calls or batched proposals with InputBatch. Measures average/max proposal latency across various ngram sizes (5-20), request counts, and token lengths. Uses TimeCollector for microsecond-precision timing.

**Significance:** Performance validation for ngram-based speculative decoding in v1 architecture. Critical for optimizing prompt lookup speculation which accelerates generation by predicting next tokens from existing context, improving throughput without accuracy loss.
