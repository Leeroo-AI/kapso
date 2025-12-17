# File: `examples/offline_inference/load_sharded_state.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 94 |
| Functions | `parse_args`, `main` |
| Imports | dataclasses, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Validates loading models saved in sharded_state format

**Mechanism:** Loads model using load_format='sharded_state' where each worker only reads its own shard rather than entire checkpoint. Uses EngineArgs with tensor_parallel_size matching saved model, runs inference to validate successful loading. Companion to save_sharded_state.py.

**Significance:** Example demonstrating fast loading path for large tensor-parallel models using per-worker sharded checkpoints.
