# File: `examples/offline_inference/rlhf.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 147 |
| Classes | `MyLLM` |
| Imports | os, ray, rlhf_utils, torch, transformers, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates RLHF training/inference separation with Ray

**Mechanism:** Separates training (GPU 0) and inference (GPUs 1-2 with tensor parallelism) using Ray placement groups. Demonstrates weight synchronization via Ray collective RPC and NCCL broadcast. Training model zeroes weights (simulating update), broadcasts to inference engine via stateless process group. References OpenRLHF for production implementation.

**Significance:** Example showing RLHF workflow pattern with separated training/inference processes and weight synchronization.
