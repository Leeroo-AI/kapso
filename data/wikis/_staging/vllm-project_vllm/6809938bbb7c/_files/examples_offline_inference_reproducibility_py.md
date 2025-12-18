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

**Purpose:** Demonstrates achieving reproducible generation outputs by controlling randomness sources.

**Mechanism:** Sets fixed seeds for Python random, environment variables, and vLLM's seed parameter. Runs multiple generations with the same seeds to show identical outputs. Uses temperature=0.8 (non-greedy) to demonstrate that stochastic sampling still produces reproducible results with proper seeding.

**Significance:** Critical for testing, debugging, and applications requiring deterministic outputs. Shows how to achieve reproducibility in vLLM despite sampling randomness and parallel execution.
