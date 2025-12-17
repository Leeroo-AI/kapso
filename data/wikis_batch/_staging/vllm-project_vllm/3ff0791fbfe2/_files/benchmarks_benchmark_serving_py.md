# File: `benchmarks/benchmark_serving.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 17 |
| Imports | sys |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Deprecation notice for serving benchmark

**Mechanism:** Prints deprecation message directing users to new vLLM CLI command (`vllm bench serving`). The script immediately exits with status 1.

**Significance:** Migration artifact preserving backward compatibility. Guides users from old standalone script to integrated CLI benchmarking commands, part of vLLM's consolidation of benchmarking tools under unified interface.
