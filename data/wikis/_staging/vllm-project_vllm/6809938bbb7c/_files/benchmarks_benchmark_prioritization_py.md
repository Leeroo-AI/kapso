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

**Purpose:** Benchmarks offline throughput with request prioritization enabled.

**Mechanism:** Samples prompts from ShareGPT dataset or generates synthetic prompts, assigns random binary priorities (0 or 1) to each request with 50% probability. Runs LLM.generate() with the priority parameter to test priority-based scheduling. Measures total elapsed time and calculates requests/second and tokens/second throughput. Supports configurable number of prompts, input/output lengths, and n parameter for multiple generations per prompt. Can optionally save results to JSON.

**Significance:** Performance validation tool for the priority scheduling feature. Allows testing whether priority-based request ordering impacts overall throughput or introduces overhead. Critical for validating that high-priority requests can be served faster without significantly degrading total system throughput. Helps quantify the tradeoff between fairness and latency for priority-sensitive workloads like interactive vs. batch serving.
