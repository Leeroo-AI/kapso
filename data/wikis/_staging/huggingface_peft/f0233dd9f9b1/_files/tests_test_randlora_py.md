# File: `tests/test_randlora.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 301 |
| Classes | `MLP`, `TestRandLora` |
| Imports | accelerate, os, peft, pytest, safetensors, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for RandLoRA adapter

**Mechanism:** Tests RandLoRA shared random projection weights across adapters with same PRNG key, ensures different adapters produce different outputs, prevents multiple adapters with different PRNG keys, and tests save/load with topk

**Significance:** Test coverage for RandLoRA random projection sharing
