# File: `benchmarks/benchmark_hash.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 120 |
| Functions | `builtin_hash`, `main` |
| Imports | __future__, argparse, collections, random, statistics, time, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Compare hash function performance

**Mechanism:** Micro-benchmarks built-in hash(), SHA-256, and xxHash on prefix-cache-shaped payloads (32-byte bytes + 32-int tuple). Runs warm-up iterations followed by timed measurement iterations, computing mean and standard deviation. Reports absolute timings and relative speedups compared to built-in hash().

**Significance:** Performance analysis tool for prefix caching hash function selection. Critical for choosing optimal hashing algorithm since prefix caching relies heavily on fast hash computation for cache key generation. Helps quantify the performance trade-offs between security (SHA-256) and speed (xxHash, built-in hash).
