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

**Purpose:** Demonstrates loading pre-sharded model weights to skip initialization and reduce startup time.

**Mechanism:** Uses LoadConfig with load_format="sharded_state" to load model weights that were previously saved in sharded format. Bypasses normal model initialization and weight distribution, directly loading pre-distributed weights from a checkpoint directory created by save_sharded_state.py.

**Significance:** Shows optimization technique for faster LLM initialization by reusing pre-sharded checkpoints. Useful for production deployments where startup time matters and weights can be pre-distributed across tensor parallel ranks.
