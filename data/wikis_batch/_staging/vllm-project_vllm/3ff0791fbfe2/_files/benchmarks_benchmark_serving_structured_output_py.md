# File: `benchmarks/benchmark_serving_structured_output.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1040 |
| Classes | `BenchmarkMetrics`, `SampleRequest` |
| Functions | `sample_requests`, `get_request`, `calculate_metrics`, `benchmark`, `evaluate`, `parse_goodput`, `check_goodput_args`, `main`, `... +1 more` |
| Imports | argparse, asyncio, backend_request_func, collections, contextlib, copy, dataclasses, datasets, json, numpy, ... +9 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmark serving with structured outputs

**Mechanism:** Comprehensive async benchmarking of structured output generation (JSON schemas, regex patterns, guided decoding). Supports multiple backends, request rate control (Poisson distribution), warmup phases, and detailed metrics (TTFT, ITL, TPOT, throughput). Calculates percentile statistics and optional goodput metrics for quality evaluation. Uses datasets like JT-NLP-Toolkit for realistic prompts.

**Significance:** Specialized benchmarking for constrained generation workloads. Critical for evaluating vLLM's structured output performance (JSON mode, regex constraints) against alternatives, essential for applications requiring strict output formats (APIs, data extraction, function calling).
