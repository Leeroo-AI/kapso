# File: `examples/offline_inference/rlhf_colocate.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 251 |
| Classes | `MyLLM`, `RayTrainingActor` |
| Imports | gc, os, ray, torch, vllm, zmq |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates RLHF with co-located training and inference

**Mechanism:** Co-locates training actors and vLLM workers on same GPUs using Ray placement groups with fractional GPU allocation (0.4 per actor/worker). Demonstrates CUDA IPC for fast in-GPU weight transfer using ZMQ sockets and reduce_tensor. Avoids NCCL limitations when multiple processes share single GPU.

**Significance:** Example showing advanced RLHF setup with training/inference on same GPUs via CUDA IPC for maximum efficiency.
