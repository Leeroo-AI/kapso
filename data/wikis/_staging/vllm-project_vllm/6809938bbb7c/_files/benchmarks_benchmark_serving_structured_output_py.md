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

**Purpose:** Benchmarks online serving throughput with structured output constraints (JSON, grammar, regex, choice).

**Mechanism:** Generates requests with structured output schemas (JSON with configurable schema files, EBNF grammar, regex patterns, or choice constraints). Supports multiple datasets including json-mode-eval (xgrammar_bench). Sends async requests at a configurable rate (Poisson or gamma distribution) with optional burstiness control and max concurrency limits. Measures TTFT, TPOT, ITL, E2EL percentiles, request throughput, output throughput, and optional goodput (requests meeting SLO). Evaluates correctness of structured outputs against expected formats. Can generate unique schemas per request to avoid caching.

**Significance:** Critical performance validation tool for structured output features. Structured outputs are essential for tool calling, JSON mode, and constrained generation use cases. Validates that grammar/schema constraints don't significantly degrade serving performance. Helps optimize the xgrammar/outlines backends for production workloads. Supports goodput metrics to measure quality-of-service under different load conditions.
