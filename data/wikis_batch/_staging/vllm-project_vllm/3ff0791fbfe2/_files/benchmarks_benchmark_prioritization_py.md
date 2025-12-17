# File: `benchmarks/benchmark_prioritization.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 221 |
| Functions | `get_random_flag`, `sample_requests`, `run_vllm`, `main`, `create_argument_parser` |
| Imports | argparse, dataclasses, json, random, time, transformers, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Test request prioritization throughput

**Mechanism:** Benchmarks offline inference with random binary priorities (0 or 1) assigned to each request. Samples prompts from ShareGPT or generates synthetic ones. Measures throughput (requests/s, tokens/s) with priority-aware scheduling. Supports JSON output for results.

**Significance:** Validation tool for priority-based request scheduling. Demonstrates vLLM's ability to handle heterogeneous request priorities (e.g., premium vs standard users) while maintaining high throughput, critical for production serving with SLA requirements.
