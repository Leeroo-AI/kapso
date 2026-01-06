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

**Purpose:** Demonstrates native data parallelism in vLLM using LLMParallelConfig to distribute inference across multiple GPUs or nodes.

**Mechanism:** Uses LLMParallelConfig with data_parallel_size to create multiple independent LLM replicas, each processing different batches of prompts. Supports both single-node multi-GPU and multi-node setups. Distributes prompts across replicas and collects results, showing throughput improvements from parallelization.

**Significance:** Shows vLLM's built-in data parallelism for scaling inference throughput without external frameworks. Critical for high-throughput batch processing scenarios where latency isn't critical but total throughput is.
