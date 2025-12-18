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

**Purpose:** Simple round-robin load balancer proxy for benchmarking multiple backend instances.

**Mechanism:** Implements aiohttp web server that forwards requests to backend services in round-robin fashion using itertools.cycle(). For each incoming request: (1) Selects next backend port from cycle, (2) Forwards request with original method, headers, and body, (3) Streams response back to client chunk-by-chunk. Handles arbitrary HTTP methods and preserves query strings. Default configuration proxies port 8000 to backends on ports 8100 and 8200.

**Significance:** Infrastructure for load distribution benchmarks. Round-robin is a simple but effective strategy for distributing load across multiple vLLM instances. Enables testing horizontal scaling scenarios where multiple model replicas serve requests. Critical for validating that load balancing doesn't introduce significant latency overhead and that throughput scales linearly with backend count. Used to simulate production deployment patterns with multiple serving instances.
