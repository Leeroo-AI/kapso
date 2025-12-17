# File: `benchmarks/benchmark_throughput.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 17 |
| Imports | sys |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Deprecation notice for throughput benchmark

**Mechanism:** Prints deprecation message directing users to new vLLM CLI command (`vllm bench throughput`). The script immediately exits with status 1.

**Significance:** Migration artifact preserving backward compatibility. Part of vLLM's transition to unified CLI interface for all benchmarking operations, improving user experience with consistent command structure.
