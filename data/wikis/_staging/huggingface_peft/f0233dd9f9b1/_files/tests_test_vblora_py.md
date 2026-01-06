# File: `tests/test_vblora.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 269 |
| Classes | `MLP`, `TestVBLoRA` |
| Imports | accelerate, os, peft, pytest, safetensors, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for VBLoRA (Vector Bank LoRA) adapter

**Mechanism:** Tests VBLoRA vector bank sharing across layers, topk vector selection, save_only_topk_weights functionality, and proper parameter shapes

**Significance:** Test coverage for VBLoRA adapter with shared vector banks
