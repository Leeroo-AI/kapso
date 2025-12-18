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

**Purpose:** Demonstrates disaggregated prefill architecture where prompt processing (prefill) and token generation (decode) run on separate vLLM instances.

**Mechanism:** Spawns two separate LLM processes: a prefill instance that processes prompts and transfers KV cache to a decode instance via RPC. Uses Ray for inter-process communication. The prefill instance outputs KV cache which the decode instance consumes to continue generation, optimizing resource utilization by separating compute-intensive prefill from memory-intensive decode.

**Significance:** Showcases advanced deployment pattern for optimizing throughput and latency by dedicating different hardware to prefill vs decode stages. Important for large-scale production deployments wanting to maximize efficiency.
