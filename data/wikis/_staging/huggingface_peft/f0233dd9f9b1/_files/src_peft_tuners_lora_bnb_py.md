# File: `src/peft/tuners/lora/bnb.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 611 |
| Classes | `Linear8bitLt`, `Linear4bit` |
| Imports | __future__, bitsandbytes, config, layer, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** BitsAndBytes quantized LoRA

**Mechanism:** Implements LoRA layers for bitsandbytes-quantized models. Linear8bitLt handles 8-bit quantized weights, Linear4bit handles 4-bit quantized weights. Both inherit from LoraLayer and wrap quantized base layers. Forward passes dequantize base layer output on-the-fly, compute LoRA contribution (B @ A @ dropout(x)) * scaling in full precision, then add to result. Merge/unmerge operations dequantize weights, apply LoRA delta, and requantize. dispatch_bnb_8bit() and dispatch_bnb_4bit() detect quantized layers and wrap them appropriately. Supports DoRA, ALoRA, and Arrow variants.

**Significance:** Critical for memory-efficient fine-tuning of large models. Enables LoRA on quantized models where base weights are stored in 4-bit or 8-bit, dramatically reducing memory footprint while maintaining LoRA adapter quality. Essential for making billion-parameter model fine-tuning accessible on consumer hardware.
