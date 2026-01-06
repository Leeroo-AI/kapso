# File: `examples/online_serving/multi_instance_data_parallel.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 87 |
| Functions | `main` |
| Imports | asyncio, threading, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates multi-instance data parallel inference setup

**Mechanism:** Creates an AsyncLLMEngine client that connects to multiple vLLM server instances (data_parallel_size=2) coordinated through RPC. Sends requests to specific data parallel ranks, enabling load distribution across instances. Includes background logging thread using AggregatedLoggingStatLogger for performance monitoring. Requires separate headless vLLM server instances running with matching data parallel configuration.

**Significance:** Reference implementation for scaling vLLM horizontally across multiple GPU nodes. Shows proper RPC coordination, rank-specific request routing, and monitoring setup. Critical example for production deployments needing throughput beyond single-instance capacity. Demonstrates the coordination handshake required between client and multiple server instances.
