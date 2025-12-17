# File: `examples/offline_inference/rlhf_utils.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 168 |
| Classes | `WorkerExtension`, `FlattenedTensorMetadata`, `ColocateWorkerExtension` |
| Functions | `stateless_init_process_group`, `rebuild_ipc` |
| Imports | collections, gc, torch, typing, zmq |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides utilities for RLHF examples

**Mechanism:** Contains WorkerExtension class for weight updates via NCCL (init_weight_update_group, update_weight, check_weights_changed), ColocateWorkerExtension for CUDA IPC-based updates (update_weights_from_ipc using ZMQ), and stateless_init_process_group helper for creating StatelessProcessGroup with PyNcclCommunicator.

**Significance:** Utility module providing worker extension classes and helpers for RLHF weight synchronization patterns.
