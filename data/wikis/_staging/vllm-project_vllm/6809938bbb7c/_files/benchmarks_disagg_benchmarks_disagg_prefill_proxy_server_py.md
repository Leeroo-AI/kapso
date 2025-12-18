# File: `benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 260 |
| Functions | `parse_args`, `main` |
| Imports | aiohttp, argparse, asyncio, logging, os, quart, time, urllib, uuid |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Proxy server for benchmarking prefill/decode disaggregation architecture.

**Mechanism:** Implements a Quart-based async proxy that splits requests between separate prefill and decode services. For each incoming request: (1) Modifies request to set max_tokens=1 and forwards to prefill service to compute KV cache, (2) Encodes KV transfer addresses (prefill and decode) into custom request_id format, (3) After prefill completes, forwards original request to decode service which streams the response back to client. Uses X-Request-Id and X-KV-Target headers to coordinate P2P KV cache transfer between services. Supports configurable timeouts, service URLs, and KV transfer ports.

**Significance:** Performance validation tool for disaggregated serving architecture where prefill and decode run on separate GPU pools. Disaggregation enables better resource utilization by matching compute requirements to workload phases - prefill needs high parallelism while decode benefits from lower latency. Critical for benchmarking KV cache transfer overhead and end-to-end latency compared to unified serving. Helps validate that disaggregation improves throughput despite additional coordination overhead.
