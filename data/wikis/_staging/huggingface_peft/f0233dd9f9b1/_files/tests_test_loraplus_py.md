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

**Purpose:** Tests for LoRA+ optimizer

**Mechanism:** Tests create_loraplus_optimizer for proper learning rate ratio setup between lora_A and lora_B matrices, embedding layers, and other parameters with bitsandbytes optimizer

**Significance:** Test coverage for LoRA+ training optimization strategy
