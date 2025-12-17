# File: `tests/test_loraplus.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 99 |
| Classes | `SimpleNet` |
| Functions | `test_lora_plus_helper_sucess`, `test_lora_plus_optimizer_sucess` |
| Imports | __future__, peft, testing_utils, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for LoRA+ optimizer with differential learning rates.

**Mechanism:** Tests the `create_loraplus_optimizer` helper function which creates an optimizer with different learning rates for lora_A, lora_B, embedding, and non-LoRA parameters. Verifies correct creation of 4 parameter groups with appropriate learning rates based on loraplus_lr_ratio and loraplus_lr_embedding settings. Tests integration with bitsandbytes 8-bit Adam.

**Significance:** Validates LoRA+ training strategy which improves convergence by using higher learning rates for lora_B matrices compared to lora_A, as proposed in the LoRA+ paper.
