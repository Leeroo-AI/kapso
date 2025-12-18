# File: `examples/offline_inference/save_sharded_state.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 87 |
| Functions | `parse_args`, `main` |
| Imports | dataclasses, os, pathlib, shutil, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates saving pre-sharded model checkpoints for faster loading in multi-GPU deployments.

**Mechanism:** Loads a model with tensor parallelism, then saves the distributed weights using save_sharded_state(). Creates checkpoint directory with per-rank weight files ready for fast loading. Eliminates need for resharding on subsequent loads.

**Significance:** Paired with load_sharded_state.py to optimize startup time in production deployments. Pre-sharding weights once enables instant loading across multiple startup cycles, important for services requiring fast restart times.
