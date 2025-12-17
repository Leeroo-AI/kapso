# File: `tests/test_boft.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 84 |
| Classes | `TestBoft` |
| Imports | peft, safetensors, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for BOFT (Butterfly Orthogonal Fine-Tuning) implementation.

**Mechanism:** Contains `TestBoft` class with two test methods: `test_boft_state_dict` verifies that the boft_P buffer is not stored in checkpoints (issue #2050) and that models load correctly without it; `test_boft_old_checkpoint_including_boft_P` ensures backward compatibility with old checkpoints that still contain the boft_P buffer.

**Significance:** Validates the BOFT checkpoint format change that made boft_P non-persistent, reducing checkpoint size while maintaining correctness and backward compatibility with existing models.
