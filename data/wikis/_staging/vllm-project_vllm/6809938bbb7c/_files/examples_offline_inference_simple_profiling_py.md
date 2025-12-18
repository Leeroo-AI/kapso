# File: `examples/offline_inference/simple_profiling.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 52 |
| Functions | `main` |
| Imports | time, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Basic profiling example showing how to measure vLLM inference latency for benchmarking.

**Mechanism:** Wraps LLM.generate() with simple time.time() measurements to capture total inference duration. Runs multiple prompts and reports timing statistics. Demonstrates simple performance measurement without requiring external profiling tools.

**Significance:** Provides straightforward pattern for basic performance benchmarking and latency measurement. Useful starting point for understanding vLLM performance characteristics before moving to more sophisticated profiling tools.
