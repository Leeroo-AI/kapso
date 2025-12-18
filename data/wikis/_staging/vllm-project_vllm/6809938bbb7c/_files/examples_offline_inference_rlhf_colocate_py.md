# File: `examples/offline_inference/rlhf_colocate.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 251 |
| Classes | `MyLLM`, `RayTrainingActor` |
| Imports | gc, os, ray, torch, vllm, zmq |

## Understanding

**Status:** ✅ Explored

**Purpose:** Advanced RLHF example demonstrating co-location of inference and training on the same GPUs for memory efficiency.

**Mechanism:** Uses ColocateWorkerExtension to enable parameter sharing between vLLM inference and PyTorch training processes on the same GPU. Implements ZMQ-based IPC for transferring tensors between processes. Alternates between generation and training phases with explicit synchronization.

**Significance:** Shows memory-optimized RLHF pattern where inference and training share GPU memory instead of duplicating models. Critical for large models where memory constraints prevent running separate instances.
