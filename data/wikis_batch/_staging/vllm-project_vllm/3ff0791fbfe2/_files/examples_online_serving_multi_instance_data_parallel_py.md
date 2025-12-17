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

**Purpose:** Data parallel multi-instance inference example

**Mechanism:** Demonstrates running multiple vLLM instances in data parallel mode by creating an AsyncLLMEngine client that connects to a distributed vLLM deployment. Sends requests to specific data parallel ranks and includes background logging. Requires coordination between multiple vLLM server instances via RPC.

**Significance:** Example showing how to use vLLM's data parallelism feature to distribute inference workload across multiple GPU instances. Important for scaling inference throughput beyond single-instance capacity.
