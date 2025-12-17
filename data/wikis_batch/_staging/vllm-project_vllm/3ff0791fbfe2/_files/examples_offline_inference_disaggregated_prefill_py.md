# File: `examples/offline_inference/disaggregated_prefill.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 127 |
| Functions | `run_prefill`, `run_decode`, `main` |
| Imports | multiprocessing, os, time, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates disaggregated prefill pattern

**Mechanism:** Launches two vLLM instances on separate GPUs: prefill node (GPU 0) handles prompt processing and KV cache generation, decode node (GPU 1) receives KV cache via P2pNcclConnector and performs token generation. Uses multiprocessing Events for coordination and KVTransferConfig for inter-instance communication.

**Significance:** Example showing separation of prefill and decode phases across different GPU instances for optimized resource utilization.
