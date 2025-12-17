# File: `examples/offline_inference/reproducibility.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 46 |
| Functions | `main` |
| Imports | os, random, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates reproducibility configuration

**Mechanism:** Shows two approaches for deterministic results: disabling multiprocessing (VLLM_ENABLE_V1_MULTIPROCESSING=0) for deterministic scheduling, or enabling batch invariance (VLLM_BATCH_INVARIANT=1) for consistency regardless of batching. Verifies random state is shared with user code.

**Significance:** Example showing how to achieve reproducible outputs despite vLLM's parallel execution.
