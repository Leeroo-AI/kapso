# File: `examples/offline_inference/torchrun_dp_example.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 151 |
| Functions | `parse_args` |
| Imports | argparse, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates data parallelism with torchrun

**Mechanism:** Uses torchrun for external launcher with distributed_executor_backend='external_launcher'. Each rank processes subset of prompts based on dp_rank and dp_size. Supports configurable TP/PP/DP/EP settings. Shows manual prompt distribution and provides tips for distributed communication.

**Significance:** Example showing torchrun-based data parallel inference without vLLM's internal process management.
