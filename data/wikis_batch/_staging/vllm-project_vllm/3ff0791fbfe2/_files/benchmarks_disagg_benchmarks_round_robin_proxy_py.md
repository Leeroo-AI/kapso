# File: `benchmarks/disagg_benchmarks/round_robin_proxy.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 63 |
| Classes | `RoundRobinProxy` |
| Functions | `main` |
| Imports | aiohttp, asyncio, itertools |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Simple round-robin load balancer

**Mechanism:** Aiohttp-based proxy cycling through target ports using itertools.cycle. Forwards all HTTP methods/paths, streams responses back to clients. Runs on configurable port (default 8000) with hardcoded backends (ports 8100, 8200).

**Significance:** Basic load distribution tool for multi-instance benchmarking. Enables testing horizontal scaling by evenly distributing requests across multiple vLLM instances, useful for baseline throughput measurements.
