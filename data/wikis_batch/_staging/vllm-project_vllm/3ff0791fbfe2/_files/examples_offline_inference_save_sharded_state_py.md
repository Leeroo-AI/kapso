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

**Purpose:** Saves model weights in per-worker sharded format

**Mechanism:** Uses llm.llm_engine.engine_core.save_sharded_state() to save each worker's state dict separately with configurable filename pattern and max file size. Copies model metadata (config, tokenizer files) to output directory while excluding original weight files. Enables fast loading via load_sharded_state.py.

**Significance:** Utility demonstrating how to save tensor-parallel models as per-worker shards for efficient loading.
