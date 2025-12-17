# File: `examples/offline_inference/data_parallel.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 268 |
| Functions | `parse_args`, `main` |
| Imports | os, time, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates data parallel inference across multiple GPUs

**Mechanism:** Distributes prompts across DP ranks using environment variables (VLLM_DP_RANK, VLLM_DP_SIZE, etc.). Each rank processes a subset of prompts with independent sampling params. Supports single-node and multi-node setups with configurable tensor/expert parallelism, quantization, and compilation options. Uses multiprocessing to spawn separate processes per DP rank.

**Significance:** Example demonstrating vLLM's data parallelism for distributing inference workload across multiple GPUs/nodes with load balancing.
