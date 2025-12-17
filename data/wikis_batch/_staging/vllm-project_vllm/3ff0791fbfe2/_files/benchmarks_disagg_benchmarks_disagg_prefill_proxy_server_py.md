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

**Purpose:** Proxy server for disaggregated prefill/decode

**Mechanism:** Quart-based async proxy coordinating prefill and decode services. Handles /v1/completions requests by splitting into two stages: prefill phase (max_tokens=1) to populate KV cache, then decode phase for full generation. Encodes KV transfer addresses in request IDs, manages streaming responses, and handles error cases. Configurable timeouts, ports, and KV transfer endpoints.

**Significance:** Reference implementation for disaggregated prefill/decode architecture benchmarking. Demonstrates how to separate compute-intensive prefill from latency-sensitive decode, enabling independent scaling of each stage for improved resource utilization.
