# File: `tests/test_lorafa.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 152 |
| Classes | `SimpleNet` |
| Functions | `test_lorafa_init_default`, `test_lorafa_init_rslora`, `test_LoraFAOptimizer_step` |
| Imports | __future__, math, peft, testing_utils, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for LoRA-FA (Frozen-A) optimizer implementation.

**Mechanism:** Tests the `create_lorafa_optimizer` function which creates an optimizer that freezes lora_A weights and only trains lora_B weights. Verifies correct scaling factor computation (both standard and RSLoRA), proper gradient freezing of lora_A, and that lora_B weights update during training steps while lora_A remains unchanged.

**Significance:** Validates the LoRA-FA training strategy which reduces memory requirements and training time by freezing half of the LoRA parameters while maintaining performance.
