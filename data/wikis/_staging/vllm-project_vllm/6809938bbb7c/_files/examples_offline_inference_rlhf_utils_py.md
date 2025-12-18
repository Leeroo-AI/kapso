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

**Purpose:** Utility module providing worker extensions for RLHF examples, enabling parameter access and IPC.

**Mechanism:** Defines WorkerExtension for exposing vLLM worker model parameters to external training processes. ColocateWorkerExtension adds ZMQ-based IPC for tensor transfer between co-located processes. Provides tensor flattening/unflattening and process group initialization utilities.

**Significance:** Foundational utilities supporting RLHF workflows. Bridges vLLM's inference-optimized architecture with training frameworks by exposing internal state and enabling efficient data transfer.
