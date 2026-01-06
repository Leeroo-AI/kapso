# File: `tests/test_boft.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 84 |
| Classes | `TestBoft` |
| Imports | peft, safetensors, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for BOFT (Butterfly Orthogonal Fine-Tuning) adapter

**Mechanism:** Tests BOFT state dict handling ensuring boft_P buffer is not persisted in checkpoints, and verifies backward compatibility with old checkpoints that include boft_P

**Significance:** Test coverage for BOFT adapter checkpoint optimization
