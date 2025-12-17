# File: `benchmarks/benchmark_latency.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 17 |
| Imports | sys |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Deprecation notice for latency benchmark

**Mechanism:** Prints deprecation message directing users to new vLLM CLI command (`vllm bench latency`). The script immediately exits with status 1.

**Significance:** Migration artifact preserving backward compatibility. Guides users from old standalone script to integrated CLI benchmarking commands, part of vLLM's effort to consolidate benchmarking tools under unified CLI interface.
