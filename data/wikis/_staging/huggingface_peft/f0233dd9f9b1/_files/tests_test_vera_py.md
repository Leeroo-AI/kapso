# File: `tests/test_vera.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 298 |
| Classes | `MLP`, `TestVera` |
| Imports | accelerate, os, peft, pytest, safetensors, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for VeRA (Vector-based Random Matrix Adaptation) adapter

**Mechanism:** Tests VeRA shared random projection weights (vera_A/vera_B) across adapters with same PRNG key, prevents different PRNG keys, tests save/load, and topk functionality

**Significance:** Test coverage for VeRA adapter with shared random projections
