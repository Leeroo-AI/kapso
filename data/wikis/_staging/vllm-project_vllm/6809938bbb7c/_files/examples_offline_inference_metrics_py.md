# File: `examples/offline_inference/metrics.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 50 |
| Functions | `main` |
| Imports | vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Shows how to collect and access vLLM's internal performance metrics after inference.

**Mechanism:** Runs inference with LLM.generate() then accesses llm.get_metrics() to retrieve detailed statistics including prompt throughput, generation throughput, time to first token (TTFT), time per output token (TPOT), and other performance counters. Displays formatted metrics output.

**Significance:** Essential for performance monitoring and optimization. Shows how to programmatically access vLLM's metrics for benchmarking, profiling, and production monitoring of inference performance.
