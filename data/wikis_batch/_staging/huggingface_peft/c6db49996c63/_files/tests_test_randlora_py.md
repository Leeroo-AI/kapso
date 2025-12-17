# File: `tests/test_randlora.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 301 |
| Classes | `MLP`, `TestRandLora` |
| Imports | accelerate, os, peft, pytest, safetensors, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for RandLora shared weight projection method

**Mechanism:** Validates RandLora's shared random projection weights (randlora_A, randlora_B) across adapters with same PRNG key, tests memory sharing behavior, prevents mixing different PRNG keys, validates save/load with and without projection weights, and tests adapter-specific lambda/gamma parameters with different layer shapes and dtypes

**Significance:** Ensures RandLora correctly implements memory-efficient adapter parameter sharing through deterministic random projections, reducing memory overhead when using multiple adapters
