# File: `examples/offline_inference/mlpspeculator.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 72 |
| Functions | `time_generation`, `main` |
| Imports | gc, time, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates MLP speculative decoding (deprecated)

**Mechanism:** Compares generation performance with and without speculative decoding using MLPSpeculator. Note: marked as out of date and not supported in vLLM v1. Measures time per token with warmup runs.

**Significance:** Deprecated example showing legacy speculative decoding approach with MLP predictor.
